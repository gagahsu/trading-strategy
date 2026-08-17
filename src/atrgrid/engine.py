"""ATR 自適應網格引擎。

每天 13:00 對每一檔標的做這件事：

1. 用昨日以前的**完整日 K** 算 Wilder ATR(14)（今天的盤還沒收，不能拿來算）。
2. 步長 ``step = clamp(k * ATR, min_step_pct * P, max_step_pct * P)``。
   波動放大，格子自動變寬；波動收斂，格子自動變窄。
3. 現價 P 與錨點 anchor 相差幾個步長，就是要成交幾份。
   跌破 → 買；漲過 → 賣。
4. 一連串風控閘門逐一檢查，全過才輸出下單建議。
5. 成交後錨點移動到新的網格價位，階數同步更新。

「一份」永遠是「手續費剛好 1 元的最大股數」，隨當日價格重算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Sequence

from .config import GridParams, Holding, Settings, resolve_params
from .fees import max_shares_for_min_fee, split_buy_cost, split_sell_cost
from .indicators import Bar, ema, wilder_atr
from .state import Position, State, Trade

# 決策動作
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
REVIEW = "REVIEW"  # 有訊號但需要人工判斷
SKIP = "SKIP"  # 資料或設定不足，無法評估


@dataclass
class Decision:
    """單一標的的當日決策。"""

    ticker: str
    name: str
    asset_class: str
    action: str
    shares: int = 0
    rungs: int = 0
    lot_shares: int = 0
    price: float = 0.0
    anchor_before: float = 0.0
    anchor_after: float = 0.0
    step: float = 0.0
    atr: float | None = None
    atr_pct: float | None = None
    step_pct: float = 0.0
    rung_before: int = 0
    rung_after: int = 0
    position_shares: int = 0
    est_gross: float = 0.0
    est_fee: int = 0
    est_tax: int = 0
    est_cash_flow: float = 0.0  # 負數表示現金流出
    est_realized_pnl: float | None = None
    signal_rungs: int = 0  # 未受限制前的原始訊號份數
    reasons: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.action in (BUY, SELL) and self.shares > 0


def lot_size(price: float, settings: Settings) -> int:
    """當日「一份」的股數。"""
    return max_shares_for_min_fee(price, settings.fee_discount, settings.fee_minimum)


def grid_step(price: float, atr: float, params: GridParams) -> float:
    """ATR 決定的網格步長，夾在百分比上下限之間。"""
    raw = params.atr_multiplier * atr
    low = params.min_step_pct / 100.0 * price
    high = params.max_step_pct / 100.0 * price
    return min(max(raw, low), high)


def apply_ex_dividends(
    position: Position, holding: Holding, today: str
) -> list[str]:
    """把已經發生、但還沒反映到錨點的除息扣掉。

    這是網格用在配息型 ETF 上最容易爆的地雷：除息當天價格自然下跌，網格若沒
    調整錨點，就會把「除息」誤判成「下跌」而觸發買進訊號。使用者持有的高股息
    與債券 ETF 大多月配或季配，這個修正是必要的。
    """
    notes: list[str] = []
    for entry in holding.ex_dividends:
        ex_date = str(entry["date"])
        if ex_date in position.applied_ex_dividends:
            continue
        if ex_date > today:
            continue
        amount = float(entry["amount"])
        position.anchor -= amount
        position.applied_ex_dividends.append(ex_date)
        notes.append(f"{ex_date} 除息 {amount:.4f} 元，錨點下調至 {position.anchor:.4f}")
    return notes


def apply_anchor_drift(
    position: Position,
    trend_ref: float | None,
    params: GridParams,
    today: str | None = None,
) -> str | None:
    """讓錨點緩慢往趨勢均線靠攏。

    純網格在單邊行情中會失效：一路漲會賣光後再也買不回，一路跌會一直接刀。
    讓錨點以 ``drift_beta`` 的比例往長期均線靠，網格中樞就會跟著趨勢移動。

    預設 ``up_only``：只允許往上漂。這樣多頭時網格會跟上，不會早早賣光；空頭時
    錨點不下移，避免系統性地「賣在低點」。

    漂移一天只做一次 —— 同一天重複執行 advise 不該把錨點推得更遠。
    """
    if params.drift_mode == "off" or trend_ref is None:
        return None
    if today is not None and position.last_drift_date == today:
        return None
    delta = params.drift_beta * (trend_ref - position.anchor)
    if params.drift_mode == "up_only" and delta <= 0:
        return None
    if abs(delta) < 1e-9:
        return None
    before = position.anchor
    position.anchor += delta
    position.last_drift_date = today
    return (
        f"錨點跟隨 EMA{params.trend_ema_period}（{trend_ref:.2f}）由 "
        f"{before:.4f} 漂移至 {position.anchor:.4f}"
    )


def _staleness_days(last_bar_date: str, today: str) -> int:
    last = datetime.strptime(last_bar_date, "%Y-%m-%d").date()
    now = datetime.strptime(today, "%Y-%m-%d").date()
    return (now - last).days


def evaluate(
    holding: Holding,
    position: Position,
    bars: Sequence[Bar],
    price: float,
    settings: Settings,
    state: State,
    today: str | None = None,
) -> Decision:
    """對單一標的產生今日決策。

    ``bars`` 必須是**已收盤**的日 K（不含今天），``price`` 是 13:00 的即時價。
    本函式會就地更新 ``position`` 的錨點（除息與漂移），但不會執行成交 ──
    成交由 :func:`commit` 負責。
    """
    today = today or date.today().isoformat()
    params = resolve_params(settings, holding)
    decision = Decision(
        ticker=holding.ticker,
        name=holding.name,
        asset_class=holding.asset_class,
        action=SKIP,
        price=price,
        anchor_before=position.anchor,
        anchor_after=position.anchor,
        rung_before=position.rung,
        rung_after=position.rung,
        position_shares=position.shares,
    )

    # ------------------------------------------------------------ 前置檢查
    if not holding.ticker_verified:
        decision.blocks.append("股票代號尚未驗證，先跑 `atrgrid verify-tickers`")
        return decision

    if price <= 0:
        decision.blocks.append("取不到有效即時價")
        return decision

    if not bars:
        decision.blocks.append("沒有日 K 資料")
        return decision

    stale = _staleness_days(bars[-1].date, today)
    if stale > settings.max_data_staleness_days:
        decision.blocks.append(
            f"日 K 資料已 {stale} 天未更新（最後一筆 {bars[-1].date}）"
        )
        return decision

    atr = wilder_atr(bars, params.atr_period)
    if atr is None or atr <= 0:
        decision.blocks.append(
            f"ATR({params.atr_period}) 無法計算，僅有 {len(bars)} 根日 K"
        )
        return decision

    # ------------------------------------------------ 錨點維護（除息、漂移）
    decision.notes.extend(apply_ex_dividends(position, holding, today))
    closes = [bar.close for bar in bars]
    trend_ref = ema(closes, params.trend_ema_period)
    drift_note = apply_anchor_drift(position, trend_ref, params, today)
    if drift_note:
        decision.notes.append(drift_note)
    decision.anchor_before = position.anchor
    decision.anchor_after = position.anchor

    # ---------------------------------------------------------------- 網格
    step = grid_step(price, atr, params)
    lot = lot_size(price, settings)
    decision.atr = atr
    decision.atr_pct = atr / price * 100
    decision.step = step
    decision.step_pct = step / price * 100
    decision.lot_shares = lot

    prev_close = bars[-1].close
    gap_atr = abs(price - prev_close) / atr
    if gap_atr > params.gap_atr_limit:
        decision.action = REVIEW
        decision.reasons.append(
            f"相對前收 {prev_close:.2f} 跳空 {gap_atr:.1f} 倍 ATR，超過 "
            f"{params.gap_atr_limit:g} 倍上限"
        )
        decision.blocks.append("異常跳空，請先確認是否為除息、拆分或重大事件")
        return decision

    distance = price - position.anchor
    raw_rungs = int(abs(distance) // step)
    if raw_rungs == 0:
        decision.action = HOLD
        decision.reasons.append(
            f"現價距錨點 {abs(distance):.3f} 元，未滿一格 {step:.3f} 元"
            f"（{decision.step_pct:.2f}%）"
        )
        return decision

    side = SELL if distance > 0 else BUY
    decision.signal_rungs = raw_rungs
    decision.action = side
    decision.reasons.append(
        f"ATR({params.atr_period})={atr:.3f}（{decision.atr_pct:.2f}%），"
        f"步長 {step:.3f} 元（{decision.step_pct:.2f}%）"
    )
    decision.reasons.append(
        f"現價 {price:.2f} {'高於' if side == SELL else '低於'}錨點 "
        f"{position.anchor:.2f} 共 {raw_rungs} 格"
    )

    # ------------------------------------------------------------ 風控閘門
    rungs = min(raw_rungs, params.max_rungs_per_day)
    if rungs < raw_rungs:
        decision.notes.append(
            f"單日上限 {params.max_rungs_per_day} 份，本次由 {raw_rungs} 份縮減"
        )

    if side == BUY:
        rungs = _limit_buy(decision, position, params, settings, state, price, lot, rungs)
    else:
        rungs = _limit_sell(decision, position, params, settings, price, lot, rungs)

    if rungs <= 0:
        decision.action = HOLD if not decision.blocks else REVIEW
        decision.shares = 0
        return decision

    shares = lot * rungs
    decision.rungs = rungs
    decision.shares = shares
    if rungs > 1:
        # 一份的定義就是手續費 1 元的上限，合併下單會踩過門檻。
        decision.notes.append(
            f"請分 {rungs} 筆、每筆 {lot} 股下單（合併成 {shares} 股一筆會多付手續費）"
        )

    if side == BUY:
        cost = split_buy_cost(
            rungs, lot, price, settings.fee_discount, settings.fee_minimum
        )
        decision.est_gross = float(cost.gross)
        decision.est_fee = cost.fee
        decision.est_cash_flow = -float(cost.net)
        decision.anchor_after = position.anchor - step * rungs
        decision.rung_after = position.rung + rungs
    else:
        cost = split_sell_cost(
            rungs,
            lot,
            price,
            holding.asset_class,
            settings.fee_discount,
            settings.fee_minimum,
        )
        decision.est_gross = float(cost.gross)
        decision.est_fee = cost.fee
        decision.est_tax = cost.tax
        decision.est_cash_flow = float(cost.proceeds)
        basis = position.peek_sell_basis(shares)
        decision.est_realized_pnl = float(cost.proceeds) - basis
        decision.anchor_after = position.anchor + step * rungs
        decision.rung_after = position.rung - rungs

    return decision


def _limit_buy(
    decision: Decision,
    position: Position,
    params: GridParams,
    settings: Settings,
    state: State,
    price: float,
    lot: int,
    rungs: int,
) -> int:
    """把買進份數壓到部位上限與現金允許的範圍內。"""
    room = params.max_buy_rungs - position.rung
    if room <= 0:
        decision.blocks.append(
            f"已加碼至上限（第 {position.rung} 階 / 上限 {params.max_buy_rungs} 階）"
        )
        return 0
    if rungs > room:
        decision.notes.append(f"受部位上限限制，由 {rungs} 份縮減為 {room} 份")
        rungs = room

    spendable = state.cash - settings.cash_floor
    if spendable <= 0:
        decision.blocks.append(
            f"可用現金 {state.cash:,.0f} 元已達保留水位 {settings.cash_floor:,.0f} 元"
        )
        return 0

    while rungs > 0:
        cost = split_buy_cost(
            rungs, lot, price, settings.fee_discount, settings.fee_minimum
        )
        if float(cost.net) <= spendable:
            break
        rungs -= 1
    if rungs == 0:
        one_lot = float(
            split_buy_cost(
                1, lot, price, settings.fee_discount, settings.fee_minimum
            ).net
        )
        decision.blocks.append(
            f"現金不足，買一份需約 {one_lot:,.0f} 元，可動用 {spendable:,.0f} 元"
        )
    return rungs


def _limit_sell(
    decision: Decision,
    position: Position,
    params: GridParams,
    settings: Settings,
    price: float,
    lot: int,
    rungs: int,
) -> int:
    """把賣出份數壓到部位下限、實際持股與「不賠售」規則之內。"""
    room = params.max_sell_rungs + position.rung
    if room <= 0:
        decision.blocks.append(
            f"已減碼至下限（第 {position.rung} 階 / 下限 -{params.max_sell_rungs} 階）"
        )
        return 0
    if rungs > room:
        decision.notes.append(f"受部位下限限制，由 {rungs} 份縮減為 {room} 份")
        rungs = room

    max_by_shares = position.shares // lot if lot else 0
    if max_by_shares < rungs:
        decision.notes.append(
            f"持股 {position.shares} 股僅夠賣 {max_by_shares} 份"
        )
        rungs = max_by_shares
    if rungs <= 0:
        decision.blocks.append(f"持股 {position.shares} 股不足一份（{lot} 股）")
        return 0

    if not params.allow_loss_sell:
        while rungs > 0:
            shares = lot * rungs
            cost = split_sell_cost(
                rungs,
                lot,
                price,
                decision.asset_class,
                settings.fee_discount,
                settings.fee_minimum,
            )
            if float(cost.proceeds) - position.peek_sell_basis(shares) > 0:
                break
            rungs -= 1
        if rungs == 0:
            decision.blocks.append(
                "扣除手續費與證交稅後為虧損賣出，已依 allow_loss_sell=false 擋下"
            )
    return rungs


def commit(state: State, decision: Decision, trade_date: str | None = None) -> None:
    """把決策寫進狀態：更新持股、錨點、階數、現金與成交紀錄。"""
    if not decision.is_actionable:
        return
    trade_date = trade_date or date.today().isoformat()
    position = state.positions[decision.ticker]

    if decision.action == BUY:
        position.apply_buy(trade_date, decision.price, decision.shares, decision.rungs)
        realized = 0.0
    else:
        gross_pnl = position.apply_sell(
            trade_date, decision.price, decision.shares, decision.rungs
        )
        realized = gross_pnl - decision.est_fee - decision.est_tax

    position.anchor = decision.anchor_after
    state.cash += decision.est_cash_flow
    state.trades.append(
        Trade(
            date=trade_date,
            ticker=decision.ticker,
            action=decision.action,
            shares=decision.shares,
            price=decision.price,
            fee=decision.est_fee,
            tax=decision.est_tax,
            rungs=decision.rungs,
            realized_pnl=round(realized, 2),
        )
    )
    state.last_run_date = trade_date


def next_grid_levels(
    position: Position, step: float, count: int = 3
) -> tuple[list[float], list[float]]:
    """回傳接下來的買進與賣出價位，讓報表可以顯示「還要跌多少才買」。"""
    buys = [position.anchor - step * i for i in range(1, count + 1)]
    sells = [position.anchor + step * i for i in range(1, count + 1)]
    return buys, sells


def trading_day_hint(today: str | None = None) -> str | None:
    """週末提醒。國定假日不在此判斷 ── 交由資料來源的新鮮度檢查兜底。"""
    today = today or date.today().isoformat()
    weekday = datetime.strptime(today, "%Y-%m-%d").date().weekday()
    if weekday >= 5:
        return "今天是週末，台股休市；以下為參考試算"
    return None


def business_days_ago(days: int, today: str | None = None) -> str:
    base = (
        datetime.strptime(today, "%Y-%m-%d").date() if today else date.today()
    )
    return (base - timedelta(days=days)).isoformat()
