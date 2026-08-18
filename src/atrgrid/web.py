"""本機網頁介面。

    atrgrid serve

啟動一個只綁在 localhost 的小型 HTTP 服務，把網格系統包成可以用瀏覽器操作的
頁面：按一下抓報價、看今天要買賣幾股、成交後回填。

為什麼要有後端而不是純前端頁面？

* Yahoo Finance 不送 CORS 標頭，瀏覽器直接 fetch 會被擋；FinMind 需要 token，
  放在前端等於公開。由 Python 這一端去抓，兩個問題都不存在。
* 更重要的是：決策邏輯完全重用 :mod:`atrgrid.engine`，不必在 JavaScript 裡
  重寫一份會慢慢走樣的複本。頁面只負責顯示後端算好的結果。

只監聽 127.0.0.1，沒有帳號密碼 —— 這是單機工具，不要對外開放。
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import (
    VALID_CLASSES,
    ConfigError,
    Holding,
    Portfolio,
    Settings,
    add_ex_dividend,
    add_holding,
    load_portfolio,
    load_settings,
    resolve_params,
    set_ticker_verified,
    update_ex_dividend,
)
from .data import DataError, PriceProvider, make_provider
from .engine import BUY, SELL, Decision, evaluate, grid_step, lot_size, next_grid_levels
from .fees import brokerage_fee, split_buy_cost, split_sell_cost
from .indicators import wilder_atr
from .state import Trade, add_position, load_state, save_state

PAGE = Path(__file__).parent / "static" / "app.html"

#: 部署到公網（Render 等）時務必設這個環境變數；沒設就照舊視為單機工具，
#: 完全不擋（本機開發、pytest 都不受影響）。前端在每個 fetch 帶
#: ``X-Auth-Token`` header，見 static/app.html 開頭的 authToken 讀取邏輯。
AUTH_TOKEN_ENV = "ATRGRID_AUTH_TOKEN"


class ApiError(Exception):
    """回給前端的可預期錯誤。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------- 序列化


def decision_to_dict(decision: Decision) -> dict[str, Any]:
    return {
        "ticker": decision.ticker,
        "name": decision.name,
        "assetClass": decision.asset_class,
        "action": decision.action,
        "shares": decision.shares,
        "rungs": decision.rungs,
        "lotShares": decision.lot_shares,
        "price": round(decision.price, 2),
        "anchor": round(decision.anchor_before, 3),
        "anchorAfter": round(decision.anchor_after, 3),
        "step": round(decision.step, 3),
        "stepPct": round(decision.step_pct, 2),
        "atr": round(decision.atr, 3) if decision.atr else None,
        "atrPct": round(decision.atr_pct, 2) if decision.atr_pct else None,
        "rung": decision.rung_before,
        "rungAfter": decision.rung_after,
        "positionShares": decision.position_shares,
        "gross": round(decision.est_gross, 0),
        "fee": decision.est_fee,
        "tax": decision.est_tax,
        "cashFlow": round(decision.est_cash_flow, 0),
        "realizedPnl": (
            round(decision.est_realized_pnl, 0)
            if decision.est_realized_pnl is not None
            else None
        ),
        "reasons": decision.reasons,
        "notes": decision.notes,
        "blocks": decision.blocks,
    }


class GridService:
    """把設定、狀態與行情兜起來，供 HTTP 層呼叫。"""

    def __init__(
        self,
        config_dir: Path,
        state_path: Path,
        provider_kind: str = "auto",
        csv_dir: Path | None = None,
        months: int = 14,
        live_overrides: dict[str, float] | None = None,
    ) -> None:
        self.config_dir = config_dir
        self.state_path = state_path
        self.provider_kind = provider_kind
        self.csv_dir = csv_dir
        self.months = months
        self.live_overrides = live_overrides or {}
        self.lock = threading.Lock()
        self._provider: PriceProvider | None = None
        self._provider_kind_built: str | None = None

    # ------------------------------------------------------------ 資源
    def settings(self) -> Settings:
        return load_settings(self.config_dir / "settings.yaml")

    def portfolio(self) -> Portfolio:
        return load_portfolio(self.config_dir / "portfolio.yaml")

    def provider(self, kind: str | None = None) -> PriceProvider:
        kind = kind or self.provider_kind
        if self._provider is None or self._provider_kind_built != kind:
            market = {
                h.ticker: str(h.overrides.get("market", "")) or None
                for h in self.portfolio().holdings
            }
            self._provider = make_provider(
                kind,
                csv_dir=self.csv_dir,
                cache_dir=Path("data/cache"),
                live_overrides=self.live_overrides,
                market={k: v for k, v in market.items() if v},
            )
            self._provider_kind_built = kind
        return self._provider

    # ------------------------------------------------------------ 端點
    def snapshot(self) -> dict[str, Any]:
        """不連網，回傳目前的設定與網格狀態。"""
        settings = self.settings()
        portfolio = self.portfolio()
        state = load_state(self.state_path)

        holdings = []
        for holding in portfolio.holdings:
            position = state.positions.get(holding.ticker)
            params = resolve_params(settings, holding)
            anchor = position.anchor if position else holding.avg_cost
            lot = lot_size(anchor, settings)
            buys, sells = (
                next_grid_levels(position, anchor * params.min_step_pct / 100, 3)
                if position
                else ([], [])
            )
            holdings.append(
                {
                    "ticker": holding.ticker,
                    "name": holding.name,
                    "assetClass": holding.asset_class,
                    "verified": holding.ticker_verified,
                    "enabled": holding.enabled,
                    "shares": position.shares if position else holding.shares,
                    "avgCost": round(holding.avg_cost, 4),
                    "anchor": round(anchor, 3),
                    "rung": position.rung if position else 0,
                    "lotShares": lot,
                    "lotValue": round(lot * anchor, 0),
                    "realizedPnl": round(position.realized_pnl, 0) if position else 0,
                    "maxBuyRungs": params.max_buy_rungs,
                    "maxSellRungs": params.max_sell_rungs,
                    "nextBuy": [round(p, 2) for p in buys],
                    "nextSell": [round(p, 2) for p in sells],
                    "exDividends": holding.ex_dividends,
                }
            )

        return {
            "asOf": date.today().isoformat(),
            "decisionTime": settings.decision_time,
            "cash": round(state.cash, 0),
            "cashFloor": round(settings.cash_floor, 0),
            "feeDiscount": float(settings.fee_discount),
            "lastRunDate": state.last_run_date,
            "providerKind": self.provider_kind,
            "unverified": sum(1 for h in portfolio.holdings if not h.ticker_verified),
            "holdings": holdings,
            "trades": [
                {
                    "id": i,
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares,
                    "price": t.price,
                    "fee": t.fee,
                    "tax": t.tax,
                    "rungs": t.rungs,
                    "realizedPnl": t.realized_pnl,
                    "note": t.note,
                    "consumedLots": t.consumed_lots,
                }
                for i, t in enumerate(state.trades)
            ][-100:],
        }

    def quotes(self, kind: str | None = None) -> dict[str, Any]:
        """抓所有標的的盤中價。個別失敗不影響其他標的。"""
        provider = self.provider(kind)
        prices: dict[str, float] = {}
        errors: dict[str, str] = {}
        for holding in self.portfolio().enabled():
            try:
                prices[holding.ticker] = round(
                    provider.live_price(holding.ticker), 2
                )
            except (DataError, ValueError, KeyError) as exc:
                errors[holding.ticker] = str(exc)
        return {"prices": prices, "errors": errors, "source": kind or self.provider_kind}

    def advise(
        self,
        prices: dict[str, float] | None = None,
        kind: str | None = None,
        today: str | None = None,
        tickers: list[str] | None = None,
    ) -> dict[str, Any]:
        """產生今日決策。``prices`` 有給就用給的，沒給的才去抓。

        ``tickers`` 有給就只算這些檔（給前端逐檔進度條用，見
        static/app.html 的 calc()），不影響決策邏輯本身 —— 跟一次算全部
        是同一套 evaluate()，只是跑的子集不同。
        """
        settings = self.settings()
        portfolio = self.portfolio()
        state = load_state(self.state_path)
        provider = self.provider(kind)
        today = today or date.today().isoformat()
        prices = prices or {}
        wanted = set(tickers) if tickers else None

        decisions: list[dict[str, Any]] = []
        for holding in portfolio.enabled():
            if wanted is not None and holding.ticker not in wanted:
                continue
            position = state.positions.get(holding.ticker)
            if position is None:
                continue
            try:
                bars = provider.daily_bars(holding.ticker, months=self.months)
                price = prices.get(holding.ticker)
                if price is None:
                    price = provider.live_price(holding.ticker)
            except (DataError, ValueError, KeyError) as exc:
                decisions.append(
                    decision_to_dict(
                        Decision(
                            ticker=holding.ticker,
                            name=holding.name,
                            asset_class=holding.asset_class,
                            action="SKIP",
                            anchor_before=position.anchor,
                            rung_before=position.rung,
                            position_shares=position.shares,
                            blocks=[f"資料取得失敗：{exc}"],
                        )
                    )
                )
                continue

            decisions.append(
                decision_to_dict(
                    evaluate(
                        holding, position, bars, float(price), settings, state,
                        today=today,
                    )
                )
            )

        actionable = [d for d in decisions if d["shares"] > 0]
        return {
            "asOf": today,
            "decisions": decisions,
            "summary": {
                "orders": sum(d["rungs"] for d in actionable),
                "tickers": len(actionable),
                "netCashFlow": sum(d["cashFlow"] for d in actionable),
                "cost": sum(d["fee"] + d["tax"] for d in actionable),
                "cash": round(state.cash, 0),
            },
        }

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        """回填一筆實際成交，並存檔。"""
        ticker = str(payload.get("ticker", "")).strip()
        action = str(payload.get("action", "")).upper()
        if action not in (BUY, SELL):
            raise ApiError(f"action 必須是 {BUY} 或 {SELL}")
        try:
            shares = int(payload["shares"])
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"shares 與 price 必須是數字：{exc}") from exc
        if shares <= 0 or price <= 0:
            raise ApiError("shares 與 price 必須大於 0")

        with self.lock:
            settings = self.settings()
            portfolio = self.portfolio()
            state = load_state(self.state_path)
            position = state.positions.get(ticker)
            if position is None:
                raise ApiError(f"{ticker} 不在狀態檔中", 404)
            try:
                holding = portfolio.by_ticker(ticker)
            except KeyError as exc:
                raise ApiError(f"{ticker} 不在持股清單中", 404) from exc

            if action == SELL and shares > position.shares:
                raise ApiError(
                    f"賣出 {shares} 股超過持股 {position.shares} 股"
                )

            trade_date = str(payload.get("date") or date.today().isoformat())
            lot = lot_size(price, settings)
            rungs = int(payload.get("rungs") or max(1, round(shares / lot)))
            step = float(payload.get("step") or 0)

            anchor_before = position.anchor
            rung_before = position.rung
            consumed_lots = None
            cash_flow = 0.0

            if action == BUY:
                cost = split_buy_cost(
                    rungs, shares // max(rungs, 1), price,
                    settings.fee_discount, settings.fee_minimum,
                )
                position.apply_buy(trade_date, price, shares, rungs)
                cash_flow = -float(cost.net)
                state.cash += cash_flow
                fee, tax, realized = cost.fee, 0, 0.0
                position.anchor = position.anchor - step * rungs if step else price
            else:
                cost = split_sell_cost(
                    rungs, shares // max(rungs, 1), price, holding.asset_class,
                    settings.fee_discount, settings.fee_minimum,
                )
                gross_pnl, consumed_lots = position.apply_sell(trade_date, price, shares, rungs)
                cash_flow = float(cost.proceeds)
                state.cash += cash_flow
                fee, tax = cost.fee, cost.tax
                realized = gross_pnl - fee - tax
                position.anchor = position.anchor + step * rungs if step else price

            state.trades.append(
                Trade(
                    date=trade_date, ticker=ticker, action=action, shares=shares,
                    price=price, fee=fee, tax=tax, rungs=rungs,
                    realized_pnl=round(realized, 2),
                    note=str(payload.get("note") or ""),
                    anchor_before=round(anchor_before, 4),
                    rung_before=rung_before,
                    consumed_lots=consumed_lots,
                    cash_flow=round(cash_flow, 2),
                )
            )
            state.last_run_date = trade_date
            save_state(state, self.state_path)

        return {
            "ok": True,
            "ticker": ticker,
            "anchor": round(position.anchor, 3),
            "rung": position.rung,
            "shares": position.shares,
            "cash": round(state.cash, 0),
            "realizedPnl": round(realized, 0),
            "fee": fee,
            "tax": tax,
        }

    def fetch_price(self, ticker: str, kind: str | None = None) -> dict[str, Any]:
        """單一標的的即時價 —— 給「新增標的」表單的「抓即時價」按鈕用。"""
        try:
            price = self.provider(kind).live_price(ticker)
        except DataError as exc:
            raise ApiError(f"{ticker} 取不到即時價：{exc}") from exc
        return {"ticker": ticker, "price": round(price, 2)}

    def preview_holding(
        self, ticker: str, asset_class: str, price: float, kind: str | None = None
    ) -> dict[str, Any]:
        """在還沒寫檔前，用指定價格試算 ATR/步長/一份股數。

        給表單的「試算」按鈕用：使用者輸入的買入價可能跟即時價不同，步長跟一份
        股數都是價格的函數（見 engine.grid_step、fees.lot_size），這裡必須重算，
        不能沿用抓即時價當下算好的數字。
        """
        if asset_class not in VALID_CLASSES:
            raise ApiError(f"assetClass 必須是 {sorted(VALID_CLASSES)} 之一")
        if price <= 0:
            raise ApiError("price 必須 > 0")

        settings = self.settings()
        params = settings.params_for(asset_class)
        lot = lot_size(price, settings)
        result: dict[str, Any] = {"lotShares": lot, "lotValue": round(lot * price, 0)}
        if lot == 1 and brokerage_fee(price, settings.fee_discount, settings.fee_minimum) > settings.fee_minimum:
            result["feeFloorNote"] = (
                "股價偏高，1 股的手續費就已經超過最低消費，"
                "「一份」已經是最小單位，不是算錯。"
            )
        try:
            bars = self.provider(kind).daily_bars(ticker, months=self.months)
        except DataError as exc:
            result["warning"] = f"抓不到日 K：{exc}"
            return result
        atr = wilder_atr(bars, params.atr_period)
        if atr is None:
            result["warning"] = (
                f"日 K 只有 {len(bars)} 根，不足 {params.atr_period + 1} 根算不出 ATR"
            )
            return result
        step = grid_step(price, atr, params)
        result.update(
            {
                "atr": round(atr, 3),
                "atrPct": round(atr / price * 100, 2),
                "step": round(step, 3),
                "stepPct": round(step / price * 100, 2),
            }
        )
        return result

    def add_holding(self, payload: dict[str, Any]) -> dict[str, Any]:
        """新增一檔標的：寫入 portfolio.yaml、建 state.json 部位，順便算 ATR/步長。

        跟 CLI 的 ``atrgrid add-holding`` 是同一套邏輯（見 cli.cmd_add_holding），
        ticker_verified 一律先寫 false —— 未經 verify-tickers 核對前不產生
        下單建議，這是 engine.evaluate 的硬性閘門，網頁這條路徑不能繞過去。

        ``price`` 是建檔錨點，直接用表單填的（可能是抓來的即時價，也可能是
        使用者自己輸入的實際買入價）——不在這裡重新抓一次，抓到的跟表單顯示
        的兜不起來會很難除錯。
        """
        ticker = str(payload.get("ticker", "")).strip()
        name = str(payload.get("name", "")).strip()
        asset_class = str(payload.get("assetClass", "")).strip()
        if not ticker:
            raise ApiError("ticker 必填")
        if not name:
            raise ApiError("name 必填")
        if asset_class not in VALID_CLASSES:
            raise ApiError(f"assetClass 必須是 {sorted(VALID_CLASSES)} 之一")
        try:
            shares = int(payload.get("shares") or 0)
            avg_cost = float(payload["avgCost"])
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"shares／avgCost／price 必須是數字：{exc}") from exc
        if shares < 0:
            raise ApiError("shares 不可為負")
        if avg_cost <= 0:
            raise ApiError("avgCost 必須 > 0")
        if price <= 0:
            raise ApiError("price 必須 > 0")

        with self.lock:
            portfolio_path = self.config_dir / "portfolio.yaml"
            portfolio = self.portfolio()
            if any(h.ticker == ticker for h in portfolio.holdings):
                raise ApiError(f"{ticker} 已經在 portfolio.yaml 中")

            today = date.today().isoformat()
            try:
                add_holding(
                    portfolio_path, ticker=ticker, name=name,
                    asset_class=asset_class, shares=shares, avg_cost=avg_cost,
                    tracked_since=today,
                )
            except ConfigError as exc:
                raise ApiError(str(exc)) from exc

            holding = self.portfolio().by_ticker(ticker)
            state = load_state(self.state_path)
            position = add_position(state, holding, price, as_of=today)
            save_state(state, self.state_path)

            result: dict[str, Any] = {
                "ok": True,
                "ticker": ticker,
                "anchor": round(position.anchor, 3),
                "shares": position.shares,
            }

        result.update(self.preview_holding(ticker, asset_class, price, payload.get("source")))
        return result

    def dividend_suggestions(self, ticker: str, kind: str | None = None) -> dict[str, Any]:
        """從行情來源抓股利事件，扣掉已登記的與建檔基準日以前的，回傳新發現的
        （唯讀，不寫檔）。基準日（含）以前的除息早就反映在建檔當下的市價
        裡，列出來只會誘使使用者誤點「登記」，把錨點雙重扣減（見
        Holding.tracked_since 的說明）。
        """
        try:
            holding = self.portfolio().by_ticker(ticker)
        except KeyError as exc:
            raise ApiError(f"{ticker} 不在持股清單中", 404) from exc
        known = {str(e.get("date")) for e in holding.ex_dividends}
        try:
            events = self.provider(kind).dividends(ticker)
        except DataError as exc:
            raise ApiError(str(exc)) from exc
        new = [
            e for e in events
            if e["date"] not in known
            and (holding.tracked_since is None or e["date"] > holding.tracked_since)
        ]
        return {"ticker": ticker, "new": new}

    def ex_dividend(self, payload: dict[str, Any]) -> dict[str, Any]:
        """登記一筆除息到 portfolio.yaml（不動 state.json 的錨點——那是下次
        evaluate() 才會做的事，見 engine.apply_ex_dividends）。"""
        ticker = str(payload.get("ticker", "")).strip()
        ex_date = str(payload.get("date", "")).strip()
        try:
            amount = float(payload["amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"amount 必須是數字：{exc}") from exc

        portfolio_path = self.config_dir / "portfolio.yaml"
        with self.lock:
            try:
                add_ex_dividend(portfolio_path, ticker, ex_date, amount)
            except KeyError as exc:
                raise ApiError(f"{ticker} 不在持股清單中", 404) from exc
            except ConfigError as exc:
                raise ApiError(str(exc)) from exc
            holding = self.portfolio().by_ticker(ticker)

        return {
            "ok": True,
            "ticker": ticker,
            "exDividends": holding.ex_dividends,
        }

    def ex_dividend_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """修改一筆已登記的除息（依原日期找到那一筆，改日期／金額）。"""
        ticker = str(payload.get("ticker", "")).strip()
        old_date = str(payload.get("oldDate", "")).strip()
        new_date = str(payload.get("date", "")).strip()
        try:
            amount = float(payload["amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"amount 必須是數字：{exc}") from exc

        portfolio_path = self.config_dir / "portfolio.yaml"
        with self.lock:
            try:
                update_ex_dividend(portfolio_path, ticker, old_date, new_date, amount)
            except KeyError as exc:
                raise ApiError(f"{ticker} 不在持股清單中", 404) from exc
            except ConfigError as exc:
                raise ApiError(str(exc)) from exc
            holding = self.portfolio().by_ticker(ticker)

        return {
            "ok": True,
            "ticker": ticker,
            "exDividends": holding.ex_dividends,
        }

    def _security_name_zh(self, ticker: str) -> str | None:
        """依序試 TWSE→FinMind→Yahoo，回傳第一個查到的名稱。

        跟 :meth:`provider` 用哪個來源算報價無關：TWSE／FinMind 登記的是
        中文名，Yahoo 給的是英文 longName/shortName。:meth:`verify_tickers`
        要拿中文名去跟 portfolio.yaml 的中文簡稱比對，所以固定用這個順序，
        不管使用者在畫面上選的來源是什麼 —— 選 Yahoo 也一樣會查到中文名，
        不然英文對中文永遠比對不出結果（CLAUDE.md 的已知地雷）。Yahoo 排
        最後只是當 TWSE／FinMind 都查不到時的備援。
        """
        for kind in ("twse", "finmind", "yahoo"):
            try:
                name = self.provider(kind).security_name(ticker)
            except DataError:
                continue
            if name:
                return name
        return None

    def _verify_one(self, holding: Holding) -> dict[str, Any]:
        try:
            actual = self._security_name_zh(holding.ticker)
        except DataError as exc:
            return {"ticker": holding.ticker, "name": holding.name,
                     "status": "unresolved", "actual": None, "error": str(exc)}
        if actual is None:
            return {"ticker": holding.ticker, "name": holding.name,
                     "status": "unresolved", "actual": None, "error": "查無此代號"}
        stripped = holding.name.replace(" ", "")
        ok = stripped in actual.replace(" ", "") or actual.replace(" ", "") in stripped
        return {
            "ticker": holding.ticker, "name": holding.name,
            "status": "ok" if ok else "mismatch",
            "actual": actual, "verified": holding.ticker_verified,
        }

    def verify_tickers(self, kind: str | None = None) -> dict[str, Any]:
        """核對 portfolio.yaml 全部持股的代號跟行情來源登記名稱是否相符。

        跟 CLI 的 `atrgrid verify-tickers` 同一套寬鬆比對（中文簡稱互相包含）。
        名稱一律用 :meth:`_security_name_zh` 拿中文名，不吃 ``kind`` 參數
        （保留這個參數只是為了跟其他端點簽名一致，呼叫端傳什麼都不影響
        比對結果）。這裡只回報，不自動改 ticker_verified，要靠
        :meth:`set_verified` 讓使用者自己看過再翻。
        """
        return {"results": [self._verify_one(h) for h in self.portfolio().holdings]}

    def verify_ticker(self, ticker: str) -> dict[str, Any]:
        """核對單一標的（給前端逐檔跑進度條用），邏輯跟 :meth:`verify_tickers` 同一套。"""
        try:
            holding = self.portfolio().by_ticker(ticker)
        except KeyError as exc:
            raise ApiError(f"{ticker} 不在持股清單中", 404) from exc
        return self._verify_one(holding)

    def set_verified(self, ticker: str, verified: bool) -> dict[str, Any]:
        """人工核對後，手動把某檔的 ticker_verified 翻成 true/false。"""
        with self.lock:
            portfolio_path = self.config_dir / "portfolio.yaml"
            try:
                set_ticker_verified(portfolio_path, ticker, verified)
            except ConfigError as exc:
                raise ApiError(str(exc)) from exc
        return {"ok": True, "ticker": ticker, "verified": verified}

    def dividend_scan_all(self, kind: str | None = None) -> dict[str, Any]:
        """掃全部持股的股利事件，回傳每檔尚未登記的新發現（唯讀，不寫檔）。

        跟 :meth:`dividend_suggestions` 是同一顆函式，只是這裡一次跑全部標的，
        給「掃描全部除息」按鈕用（CLI 對應 `atrgrid dividends`）。
        """
        results = []
        for holding in self.portfolio().holdings:
            try:
                found = self.dividend_suggestions(holding.ticker, kind)
            except ApiError as exc:
                results.append(
                    {"ticker": holding.ticker, "name": holding.name,
                     "new": [], "error": str(exc)}
                )
                continue
            if found["new"]:
                results.append(
                    {"ticker": holding.ticker, "name": holding.name, "new": found["new"]}
                )
        return {"results": results}

    def set_cash(self, amount: float) -> dict[str, Any]:
        with self.lock:
            state = load_state(self.state_path)
            state.cash = float(amount)
            save_state(state, self.state_path)
        return {"ok": True, "cash": round(state.cash, 0)}

    def update_trade(self, trade_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            state = load_state(self.state_path)
            if 0 <= trade_id < len(state.trades):
                t = state.trades[trade_id]
                t.date = payload.get("date", t.date)
                t.ticker = payload.get("ticker", t.ticker)
                t.action = payload.get("action", t.action)
                t.shares = int(payload.get("shares", t.shares))
                t.price = float(payload.get("price", t.price))
                t.fee = int(payload.get("fee", t.fee))
                t.tax = int(payload.get("tax", t.tax))
                t.rungs = int(payload.get("rungs", t.rungs))
                t.realized_pnl = float(payload.get("realizedPnl", t.realized_pnl))
                t.note = payload.get("note", t.note)
                save_state(state, self.state_path)
            else:
                raise ApiError("找不到該筆交易")
        return {"ok": True}

    def delete_trade(self, trade_id: int) -> dict[str, Any]:
        with self.lock:
            state = load_state(self.state_path)
            if not (0 <= trade_id < len(state.trades)):
                raise ApiError("找不到該筆交易")
                
            trade = state.trades[trade_id]
            ticker = trade.ticker
            
            # 檢查是否為該標的最後一筆交易
            for t in state.trades[trade_id+1:]:
                if t.ticker == ticker:
                    raise ApiError("只能倒帶該標的的「最後一筆」交易紀錄，因為後續已有新交易發生。")
                    
            if trade.anchor_before is None or trade.rung_before is None or trade.cash_flow is None:
                # 舊版紀錄，無倒帶資訊，只能單純刪除明細
                state.trades.pop(trade_id)
                save_state(state, self.state_path)
                return {"ok": True, "warning": "這是一筆舊版紀錄，缺乏倒帶資訊，因此僅刪除紀錄，未連動修改庫存與現金。"}

            # 執行倒帶
            position = state.positions.get(ticker)
            if position:
                state.cash -= trade.cash_flow
                position.anchor = trade.anchor_before
                position.rung = trade.rung_before
                
                if trade.action == BUY:
                    position.shares -= trade.shares
                    # 移除買進時產生的 Lot
                    if position.lots and position.lots[-1].date == trade.date and position.lots[-1].shares == trade.shares:
                        position.lots.pop()
                    else:
                        raise ApiError("庫存 Lot 狀態與交易紀錄不符，無法自動復原")
                else:
                    position.shares += trade.shares
                    position.realized_pnl -= trade.realized_pnl
                    # 把賣出消耗的 Lot 放回
                    from .state import Lot
                    for c_lot in (trade.consumed_lots or []):
                        position.lots.append(Lot(
                            date=c_lot["date"], 
                            price=c_lot["price"], 
                            shares=c_lot["shares"], 
                            source=c_lot.get("source", "grid")
                        ))
                        
            state.trades.pop(trade_id)
            save_state(state, self.state_path)
        return {"ok": True}


# ------------------------------------------------------------------- HTTP


def make_handler(service: GridService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "atrgrid"

        def log_message(self, fmt: str, *args: Any) -> None:
            # 預設的 stderr 逐筆日誌太吵，只留錯誤。
            if not str(args[1] if len(args) > 1 else "").startswith("2"):
                super().log_message(fmt, *args)

        # ---------------------------------------------------- 回應工具
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiError(f"請求不是合法的 JSON：{exc}") from exc

        def _dispatch(self, handler) -> None:
            try:
                self._json(handler())
            except ApiError as exc:
                self._json({"error": str(exc)}, exc.status)
            except (DataError, KeyError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - 最後防線
                self._json({"error": f"伺服器錯誤：{exc}"}, 500)

        # ---------------------------------------------------- 認證
        def _authorized(self) -> bool:
            required = os.environ.get(AUTH_TOKEN_ENV)
            if not required:
                return True
            given = self.headers.get("X-Auth-Token", "")
            return hmac.compare_digest(given, required)

        # ---------------------------------------------------- 路由
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                # 頁面殼本身不含資料，允許不帶 token 也能載入 —— 資料都走
                # /api/* 拿，那些才是真正要保護的東西。也讓瀏覽器能先顯示
                # 「請輸入 token」的畫面，而不是連殼都出不來。
                try:
                    body = PAGE.read_bytes()
                except OSError:
                    self._json({"error": f"找不到頁面檔案 {PAGE}"}, 500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/snapshot":
                if not self._authorized():
                    self._json({"error": "unauthorized"}, 401)
                    return
                self._dispatch(service.snapshot)
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if not self._authorized():
                self._json({"error": "unauthorized"}, 401)
                return
            if path == "/api/quotes":
                self._dispatch(lambda: service.quotes(self._body().get("source")))
            elif path == "/api/advise":
                def run():
                    body = self._body()
                    raw = body.get("prices") or {}
                    prices = {
                        str(k): float(v)
                        for k, v in raw.items()
                        if v not in (None, "", "-")
                    }
                    tickers = body.get("tickers") or None
                    return service.advise(prices, body.get("source"), tickers=tickers)
                self._dispatch(run)
            elif path == "/api/record":
                self._dispatch(lambda: service.record(self._body()))
            elif path == "/api/add-holding":
                self._dispatch(lambda: service.add_holding(self._body()))
            elif path == "/api/quote":
                def run_quote():
                    body = self._body()
                    ticker = str(body.get("ticker", "")).strip()
                    if not ticker:
                        raise ApiError("ticker 必填")
                    return service.fetch_price(ticker, body.get("source"))
                self._dispatch(run_quote)
            elif path == "/api/preview-holding":
                def run_preview():
                    body = self._body()
                    ticker = str(body.get("ticker", "")).strip()
                    if not ticker:
                        raise ApiError("ticker 必填")
                    try:
                        price = float(body.get("price"))
                    except (TypeError, ValueError) as exc:
                        raise ApiError(f"price 必須是數字：{exc}") from exc
                    return service.preview_holding(
                        ticker, str(body.get("assetClass", "")), price, body.get("source")
                    )
                self._dispatch(run_preview)
            elif path == "/api/ex-dividend":
                self._dispatch(lambda: service.ex_dividend(self._body()))
            elif path == "/api/ex-dividend-update":
                self._dispatch(lambda: service.ex_dividend_update(self._body()))
            elif path == "/api/dividend-suggestions":
                def run_div():
                    body = self._body()
                    ticker = str(body.get("ticker", "")).strip()
                    if not ticker:
                        raise ApiError("ticker 必填")
                    return service.dividend_suggestions(ticker, body.get("source"))
                self._dispatch(run_div)
            elif path == "/api/dividend-scan":
                self._dispatch(
                    lambda: service.dividend_scan_all(self._body().get("source"))
                )
            elif path == "/api/verify-tickers":
                self._dispatch(
                    lambda: service.verify_tickers(self._body().get("source"))
                )
            elif path == "/api/verify-ticker":
                def run_verify_ticker():
                    ticker = str(self._body().get("ticker", "")).strip()
                    if not ticker:
                        raise ApiError("ticker 必填")
                    return service.verify_ticker(ticker)
                self._dispatch(run_verify_ticker)
            elif path == "/api/set-verified":
                def run_set_verified():
                    body = self._body()
                    ticker = str(body.get("ticker", "")).strip()
                    if not ticker:
                        raise ApiError("ticker 必填")
                    return service.set_verified(ticker, bool(body.get("verified")))
                self._dispatch(run_set_verified)
            elif path == "/api/cash":
                self._dispatch(
                    lambda: service.set_cash(float(self._body().get("cash", 0)))
                )
            elif path == "/api/update-trade":
                def run_update_trade():
                    body = self._body()
                    return service.update_trade(int(body["id"]), body)
                self._dispatch(run_update_trade)
            elif path == "/api/delete-trade":
                self._dispatch(lambda: service.delete_trade(int(self._body()["id"])))
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(
    service: GridService,
    host: str = "127.0.0.1",
    port: int = 8770,
    open_browser: bool = True,
) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    url = f"http://{host}:{port}/"
    print(f"ATR 網格網頁介面：{url}")
    print("按 Ctrl-C 結束。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
