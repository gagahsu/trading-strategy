"""單一標的的網格回測。

用途不是「預測會賺多少」，而是**檢查參數合不合理**：一年會成交幾次？
成本吃掉多少？部位會不會撞到上下限而失效？

模擬把每天的收盤價當成 13:00 的參考價，並且只用該日之前的 K 棒算 ATR ──
與正式流程一致，避免未來資料洩漏。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import Holding, Settings, resolve_params
from .engine import BUY, commit, evaluate
from .indicators import Bar
from .state import Position, State


@dataclass
class BacktestResult:
    ticker: str
    name: str
    bars: int
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    buys: int = 0
    sells: int = 0
    shares_start: int = 0
    shares_end: int = 0
    cash_start: float = 0.0
    cash_end: float = 0.0
    total_fees: int = 0
    total_tax: int = 0
    realized_pnl: float = 0.0
    max_rung: int = 0
    min_rung: int = 0
    days_at_buy_limit: int = 0
    days_at_sell_limit: int = 0
    trades: list[dict] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return self.buys + self.sells

    @property
    def equity_end(self) -> float:
        return self.cash_end + self.shares_end * self.end_price

    @property
    def equity_start(self) -> float:
        return self.cash_start + self.shares_start * self.start_price

    @property
    def buy_and_hold_end(self) -> float:
        """完全不動、只抱著初始股數與現金的終值。"""
        return self.cash_start + self.shares_start * self.end_price

    @property
    def grid_edge(self) -> float:
        """網格相對純抱著的超額報酬（元）。"""
        return self.equity_end - self.buy_and_hold_end

    @property
    def trades_per_year(self) -> float:
        if self.bars == 0:
            return 0.0
        return self.total_trades / (self.bars / 244)

    def summary(self) -> str:
        lines = [
            f"{self.name} ({self.ticker})　{self.start_date} → {self.end_date}"
            f"　{self.bars} 個交易日",
            f"  價格　　　{self.start_price:.2f} → {self.end_price:.2f}"
            f"（{(self.end_price / self.start_price - 1) * 100:+.1f}%）",
            f"  成交　　　買 {self.buys} 次 / 賣 {self.sells} 次"
            f"（年化 {self.trades_per_year:.1f} 次）",
            f"  持股　　　{self.shares_start:,} → {self.shares_end:,} 股",
            f"  階數區間　{self.min_rung:+d} ~ {self.max_rung:+d}"
            f"（觸頂 {self.days_at_buy_limit} 日 / 觸底 {self.days_at_sell_limit} 日）",
            f"  交易成本　手續費 {self.total_fees:,} + 證交稅 {self.total_tax:,}"
            f" = {self.total_fees + self.total_tax:,} 元",
            f"  已實現　　{self.realized_pnl:+,.0f} 元",
            f"  期末權益　{self.equity_end:,.0f} 元"
            f"（純持有 {self.buy_and_hold_end:,.0f}，差額 {self.grid_edge:+,.0f}）",
        ]
        return "\n".join(lines)


def run_backtest(
    holding: Holding,
    bars: Sequence[Bar],
    settings: Settings,
    warmup: int | None = None,
    cash: float | None = None,
) -> BacktestResult:
    """重播日 K，逐日產生並執行決策。

    ``warmup`` 是開始交易前保留的 K 棒數，預設為 ATR 週期加趨勢均線長度，
    確保第一筆訊號就有完整的指標。
    """
    params = resolve_params(settings, holding)
    if warmup is None:
        warmup = max(params.atr_period + 1, params.trend_ema_period) + 5
    if len(bars) <= warmup + 1:
        raise ValueError(
            f"{holding.ticker}: 需要至少 {warmup + 2} 根 K 棒，目前只有 {len(bars)}"
        )

    start_cash = settings.cash if cash is None else cash
    start_price = bars[warmup].close

    position = Position(
        ticker=holding.ticker,
        shares=holding.shares,
        anchor=start_price,
        rung=0,
        baseline_shares=holding.shares,
        lots=[],
    )
    if holding.shares > 0:
        from .state import Lot

        position.lots.append(
            Lot(
                date=bars[warmup].date,
                price=holding.avg_cost,
                shares=holding.shares,
                source="initial",
            )
        )
    state = State(cash=start_cash, positions={holding.ticker: position})

    result = BacktestResult(
        ticker=holding.ticker,
        name=holding.name,
        bars=len(bars) - warmup,
        start_date=bars[warmup].date,
        end_date=bars[-1].date,
        start_price=start_price,
        end_price=bars[-1].close,
        shares_start=holding.shares,
        cash_start=start_cash,
    )

    for i in range(warmup + 1, len(bars)):
        history = bars[:i]  # 只有前一日之前的完整 K 棒
        today = bars[i].date
        price = bars[i].close

        decision = evaluate(
            holding=holding,
            position=position,
            bars=history,
            price=price,
            settings=settings,
            state=state,
            today=today,
        )

        if position.rung >= params.max_buy_rungs:
            result.days_at_buy_limit += 1
        if position.rung <= -params.max_sell_rungs:
            result.days_at_sell_limit += 1

        if not decision.is_actionable:
            continue

        commit(state, decision, trade_date=today)
        trade = state.trades[-1]
        if decision.action == BUY:
            result.buys += 1
        else:
            result.sells += 1
        result.total_fees += trade.fee
        result.total_tax += trade.tax
        result.realized_pnl += trade.realized_pnl
        result.max_rung = max(result.max_rung, position.rung)
        result.min_rung = min(result.min_rung, position.rung)
        result.trades.append(
            {
                "date": today,
                "action": decision.action,
                "shares": decision.shares,
                "price": round(price, 2),
                "rungs": decision.rungs,
                "rung_after": position.rung,
                "realized_pnl": trade.realized_pnl,
            }
        )

    result.shares_end = position.shares
    result.cash_end = state.cash
    return result


def sweep_multiplier(
    holding: Holding,
    bars: Sequence[Bar],
    settings: Settings,
    multipliers: Sequence[float] = (0.3, 0.5, 0.75, 1.0, 1.5),
    cash: float | None = None,
) -> list[tuple[float, BacktestResult]]:
    """掃描不同的 ATR 倍數，看步長鬆緊對成交頻率與績效的影響。"""
    from dataclasses import replace as dc_replace

    out: list[tuple[float, BacktestResult]] = []
    for k in multipliers:
        overrides = dict(holding.overrides)
        overrides["atr_multiplier"] = k
        variant = dc_replace(holding, overrides=overrides)
        try:
            out.append((k, run_backtest(variant, bars, settings, cash=cash)))
        except ValueError:
            continue
    return out
