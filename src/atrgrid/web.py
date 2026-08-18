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

import json
import threading
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import Portfolio, Settings, load_portfolio, load_settings, resolve_params
from .data import DataError, PriceProvider, make_provider
from .engine import BUY, SELL, Decision, evaluate, lot_size, next_grid_levels
from .fees import split_buy_cost, split_sell_cost
from .state import Trade, load_state, save_state

PAGE = Path(__file__).parent / "static" / "app.html"


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
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares,
                    "price": t.price,
                    "fee": t.fee,
                    "tax": t.tax,
                    "realizedPnl": t.realized_pnl,
                }
                for t in state.trades[-40:]
            ],
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
    ) -> dict[str, Any]:
        """產生今日決策。``prices`` 有給就用給的，沒給的才去抓。"""
        settings = self.settings()
        portfolio = self.portfolio()
        state = load_state(self.state_path)
        provider = self.provider(kind)
        today = today or date.today().isoformat()
        prices = prices or {}

        decisions: list[dict[str, Any]] = []
        for holding in portfolio.enabled():
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

            if action == BUY:
                cost = split_buy_cost(
                    rungs, shares // max(rungs, 1), price,
                    settings.fee_discount, settings.fee_minimum,
                )
                position.apply_buy(trade_date, price, shares, rungs)
                state.cash -= float(cost.net)
                fee, tax, realized = cost.fee, 0, 0.0
                position.anchor = position.anchor - step * rungs if step else price
            else:
                cost = split_sell_cost(
                    rungs, shares // max(rungs, 1), price, holding.asset_class,
                    settings.fee_discount, settings.fee_minimum,
                )
                gross_pnl = position.apply_sell(trade_date, price, shares, rungs)
                state.cash += float(cost.proceeds)
                fee, tax = cost.fee, cost.tax
                realized = gross_pnl - fee - tax
                position.anchor = position.anchor + step * rungs if step else price

            state.trades.append(
                Trade(
                    date=trade_date, ticker=ticker, action=action, shares=shares,
                    price=price, fee=fee, tax=tax, rungs=rungs,
                    realized_pnl=round(realized, 2),
                    note=str(payload.get("note") or ""),
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

    def set_cash(self, amount: float) -> dict[str, Any]:
        with self.lock:
            state = load_state(self.state_path)
            state.cash = float(amount)
            save_state(state, self.state_path)
        return {"ok": True, "cash": round(state.cash, 0)}


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

        # ---------------------------------------------------- 路由
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
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
                self._dispatch(service.snapshot)
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
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
                    return service.advise(prices, body.get("source"))
                self._dispatch(run)
            elif path == "/api/record":
                self._dispatch(lambda: service.record(self._body()))
            elif path == "/api/cash":
                self._dispatch(
                    lambda: service.set_cash(float(self._body().get("cash", 0)))
                )
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
