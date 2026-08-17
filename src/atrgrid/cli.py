"""指令列入口。

    atrgrid init            從 portfolio.yaml 建立初始狀態
    atrgrid verify-tickers  用交易所資料核對股票代號
    atrgrid advise          產生今日 13:00 建議（預設不改狀態）
    atrgrid record          記錄實際成交（可從 advise 的結果帶入）
    atrgrid status          顯示目前網格狀態
    atrgrid backtest        單一標的回測
    atrgrid fetch           下載日 K 存成 CSV
    atrgrid lot             查詢某價位的「一份」是幾股
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtest import run_backtest, sweep_multiplier
from .config import (
    ConfigError,
    check_step_covers_costs,
    default_config_dir,
    load_portfolio,
    load_settings,
    resolve_params,
)
from .data import DataError, PriceProvider, export_csv, make_provider
from .engine import (
    BUY,
    SELL,
    Decision,
    commit,
    evaluate,
    lot_size,
    trading_day_hint,
)
from .fees import buy_cost, max_shares_for_min_fee, round_trip_cost_pct, sell_cost
from .report import (
    ReportContext,
    render_html,
    render_markdown,
    render_state_summary,
    render_text,
    today_iso,
)
from .state import Trade, init_state, load_state, save_state

DEFAULT_STATE = Path("state/state.json")


# --------------------------------------------------------------------- 共用


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    config_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
    return (
        config_dir / "settings.yaml",
        config_dir / "portfolio.yaml",
        Path(args.state) if args.state else DEFAULT_STATE,
    )


def _build_provider(args: argparse.Namespace) -> PriceProvider:
    overrides: dict[str, float] = {}
    for item in getattr(args, "price", None) or []:
        ticker, _, value = item.partition("=")
        overrides[ticker.strip()] = float(value)
    return make_provider(
        args.provider,
        csv_dir=getattr(args, "csv_dir", None),
        cache_dir=Path("data/cache"),
        live_overrides=overrides,
    )


def _load_all(args: argparse.Namespace):
    settings_path, portfolio_path, state_path = _paths(args)
    settings = load_settings(settings_path)
    portfolio = load_portfolio(portfolio_path)
    return settings, portfolio, state_path


# --------------------------------------------------------------------- init


def cmd_init(args: argparse.Namespace) -> int:
    settings, portfolio, state_path = _load_all(args)
    if state_path.exists() and not args.force:
        print(f"狀態檔已存在：{state_path}（要重建請加 --force）", file=sys.stderr)
        return 1

    provider = _build_provider(args)
    prices: dict[str, float] = {}
    for holding in portfolio.enabled():
        try:
            prices[holding.ticker] = provider.live_price(holding.ticker)
        except DataError as exc:
            print(f"  ! {holding.ticker} {holding.name}：{exc}", file=sys.stderr)
            if args.fallback_to_cost:
                prices[holding.ticker] = holding.avg_cost
                print(f"    → 改用平均成本 {holding.avg_cost:.2f} 建檔")
            else:
                return 1

    cash = args.cash if args.cash is not None else settings.cash
    state = init_state(portfolio.enabled(), prices, cash=cash)
    save_state(state, state_path)
    print(f"已建立 {len(state.positions)} 檔標的的網格狀態 → {state_path}")
    print(f"現金池 {cash:,.0f} 元")
    print()
    print("錨點（網格從今天的價格開始）：")
    for ticker, position in sorted(state.positions.items()):
        holding = portfolio.by_ticker(ticker)
        lot = lot_size(position.anchor, settings)
        print(
            f"  {ticker:<8}{holding.name:<16}錨點 {position.anchor:>8.2f}"
            f"　一份 {lot:>5} 股（約 {lot * position.anchor:,.0f} 元）"
        )
    return 0


# ----------------------------------------------------------- verify-tickers


def cmd_verify_tickers(args: argparse.Namespace) -> int:
    _, portfolio, _ = _load_all(args)
    provider = _build_provider(args)

    print("核對股票代號與交易所登記名稱：\n")
    mismatches = 0
    unresolved = 0
    for holding in portfolio.holdings:
        try:
            actual = provider.security_name(holding.ticker)
        except DataError as exc:
            actual = None
            print(f"  ?  {holding.ticker:<8}{holding.name:<20}查詢失敗：{exc}")
            unresolved += 1
            continue
        if actual is None:
            print(f"  ?  {holding.ticker:<8}{holding.name:<20}查無此代號")
            unresolved += 1
            continue
        # 交易所名稱常有簡稱差異，用互相包含做寬鬆比對。
        stripped = holding.name.replace(" ", "")
        ok = stripped in actual.replace(" ", "") or actual.replace(" ", "") in stripped
        mark = "OK" if ok else "!!"
        if not ok:
            mismatches += 1
        print(f"  {mark}  {holding.ticker:<8}{holding.name:<20}交易所：{actual}")

    print()
    if mismatches or unresolved:
        print(
            f"有 {mismatches} 檔名稱不符、{unresolved} 檔查不到。"
            "請修正 portfolio.yaml 的 ticker 後再把 ticker_verified 設為 true。"
        )
        return 1
    print("全部相符。請把 portfolio.yaml 中各檔的 ticker_verified 設為 true。")
    return 0


# ------------------------------------------------------------------- advise


def cmd_advise(args: argparse.Namespace) -> int:
    settings, portfolio, state_path = _load_all(args)
    state = load_state(state_path)
    if not state.positions:
        print("狀態尚未建立，請先執行 `atrgrid init`", file=sys.stderr)
        return 1

    provider = _build_provider(args)
    today = args.date or today_iso()
    warnings: list[str] = []

    hint = trading_day_hint(today)
    if hint:
        warnings.append(hint)

    decisions: list[Decision] = []
    for holding in portfolio.enabled():
        position = state.positions.get(holding.ticker)
        if position is None:
            warnings.append(f"{holding.ticker} {holding.name} 不在狀態檔中，已略過")
            continue

        cost_warning = check_step_covers_costs(settings, holding, holding.avg_cost)
        if cost_warning:
            warnings.append(cost_warning)

        try:
            bars = provider.daily_bars(holding.ticker, months=args.months)
            price = provider.live_price(holding.ticker)
        except DataError as exc:
            decisions.append(
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
            continue

        decisions.append(
            evaluate(holding, position, bars, price, settings, state, today=today)
        )

    ctx = ReportContext(
        as_of=today,
        decision_time=settings.decision_time,
        state=state,
        decisions=decisions,
        warnings=warnings,
    )

    output = {
        "text": render_text,
        "markdown": render_markdown,
        "html": render_html,
    }[args.format](ctx)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"報表已寫入 {args.out}")
    else:
        print(output)

    if args.json:
        payload = [
            {
                "ticker": d.ticker,
                "name": d.name,
                "action": d.action,
                "shares": d.shares,
                "rungs": d.rungs,
                "lot_shares": d.lot_shares,
                "est_fee": d.est_fee,
                "est_tax": d.est_tax,
                "price": round(d.price, 2),
                "anchor_before": round(d.anchor_before, 4),
                "anchor_after": round(d.anchor_after, 4),
                "step": round(d.step, 4),
                "atr": round(d.atr, 4) if d.atr else None,
                "est_cash_flow": round(d.est_cash_flow, 2),
                "reasons": d.reasons,
                "blocks": d.blocks,
                "notes": d.notes,
            }
            for d in ctx.sorted_decisions()
        ]
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON 已寫入 {args.json}")

    # --commit 代表「照建議全部成交」，僅適合模擬或全自動流程。
    if args.commit:
        applied = 0
        for decision in ctx.actionable():
            commit(state, decision, trade_date=today)
            applied += 1
        state.last_run_date = today
        save_state(state, state_path)
        print(f"\n已將 {applied} 筆建議寫入狀態（{state_path}）")
    else:
        # 錨點漂移與除息調整即使沒下單也應該留存，否則每天都會重算。
        if args.persist_anchors:
            state.last_run_date = today
            save_state(state, state_path)
            print(f"\n錨點調整已存檔（{state_path}）")

    return 0


# ------------------------------------------------------------------- record


def cmd_record(args: argparse.Namespace) -> int:
    settings, portfolio, state_path = _load_all(args)
    state = load_state(state_path)
    holding = portfolio.by_ticker(args.ticker)
    position = state.positions.get(args.ticker)
    if position is None:
        print(f"{args.ticker} 不在狀態檔中", file=sys.stderr)
        return 1

    params = resolve_params(settings, holding)
    trade_date = args.date or today_iso()
    price = args.price
    shares = args.shares
    lot = lot_size(price, settings)
    rungs = args.rungs if args.rungs is not None else max(1, round(shares / lot))
    step = args.step if args.step is not None else 0.0

    if args.action == BUY:
        cost = buy_cost(shares, price, settings.fee_discount, settings.fee_minimum)
        position.apply_buy(trade_date, price, shares, rungs)
        state.cash -= float(cost.net)
        realized = 0.0
        fee, tax = cost.fee, 0
        if step:
            position.anchor -= step * rungs
        else:
            position.anchor = price
    else:
        cost = sell_cost(
            shares, price, holding.asset_class, settings.fee_discount, settings.fee_minimum
        )
        gross_pnl = position.apply_sell(trade_date, price, shares, rungs)
        state.cash += float(cost.proceeds)
        realized = gross_pnl - cost.fee - cost.tax
        fee, tax = cost.fee, cost.tax
        if step:
            position.anchor += step * rungs
        else:
            position.anchor = price

    state.trades.append(
        Trade(
            date=trade_date,
            ticker=args.ticker,
            action=args.action,
            shares=shares,
            price=price,
            fee=fee,
            tax=tax,
            rungs=rungs,
            realized_pnl=round(realized, 2),
            note=args.note or "",
        )
    )
    state.last_run_date = trade_date
    save_state(state, state_path)

    print(
        f"已記錄　{trade_date}　{args.action} {holding.name} {shares} 股 @ {price:.2f}"
    )
    print(f"　　費用 {fee} 元、稅 {tax} 元、實現損益 {realized:+,.0f} 元")
    print(f"　　新錨點 {position.anchor:.4f}　階數 {position.rung:+d}"
          f"　持股 {position.shares:,} 股　現金 {state.cash:,.0f} 元")
    if params.max_buy_rungs < position.rung or position.rung < -params.max_sell_rungs:
        print("　　⚠ 階數已超出設定範圍，請檢查參數或狀態")
    return 0


# ------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> int:
    settings, portfolio, state_path = _load_all(args)
    state = load_state(state_path)
    if not state.positions:
        print("狀態尚未建立，請先執行 `atrgrid init`", file=sys.stderr)
        return 1
    print(render_state_summary(state))

    if args.trades:
        print("\n最近成交：")
        for trade in state.trades[-args.trades :]:
            print(
                f"  {trade.date}  {trade.action:<4}{trade.ticker:<8}"
                f"{trade.shares:>7,} 股 @ {trade.price:>8.2f}"
                f"　費{trade.fee}+稅{trade.tax}　{trade.realized_pnl:+,.0f}"
            )
    return 0


# ----------------------------------------------------------------- backtest


def cmd_backtest(args: argparse.Namespace) -> int:
    settings, portfolio, _ = _load_all(args)
    provider = _build_provider(args)

    tickers = args.tickers or [h.ticker for h in portfolio.enabled()]
    results = []
    for ticker in tickers:
        try:
            holding = portfolio.by_ticker(ticker)
        except KeyError:
            print(f"{ticker} 不在 portfolio.yaml 中，略過", file=sys.stderr)
            continue
        # 回測需要繞過代號驗證閘門。
        from dataclasses import replace as dc_replace

        holding = dc_replace(holding, ticker_verified=True)
        try:
            bars = provider.daily_bars(ticker, months=args.months)
        except DataError as exc:
            print(f"{ticker}：{exc}", file=sys.stderr)
            continue

        if args.sweep:
            print(f"\n=== {holding.name} ({ticker}) ATR 倍數掃描 ===")
            print(
                f"{'k':>6}{'買':>5}{'賣':>5}{'成本':>9}{'實現':>11}"
                f"{'期末權益':>13}{'vs 純持有':>12}"
            )
            for k, result in sweep_multiplier(
                holding, bars, settings, cash=args.cash
            ):
                print(
                    f"{k:>6.2f}{result.buys:>5}{result.sells:>5}"
                    f"{result.total_fees + result.total_tax:>9,}"
                    f"{result.realized_pnl:>+11,.0f}{result.equity_end:>13,.0f}"
                    f"{result.grid_edge:>+12,.0f}"
                )
            continue

        try:
            result = run_backtest(holding, bars, settings, cash=args.cash)
        except ValueError as exc:
            print(f"{ticker}：{exc}", file=sys.stderr)
            continue
        results.append(result)
        print()
        print(result.summary())
        if args.trades:
            for trade in result.trades[-args.trades :]:
                print(
                    f"    {trade['date']}  {trade['action']:<4}"
                    f"{trade['shares']:>7,} 股 @ {trade['price']:>8.2f}"
                    f"　階 {trade['rung_after']:+d}　{trade['realized_pnl']:+,.0f}"
                )

    if len(results) > 1:
        print("\n" + "=" * 60)
        print(
            f"合計　成交 {sum(r.total_trades for r in results)} 次"
            f"　成本 {sum(r.total_fees + r.total_tax for r in results):,} 元"
            f"　相對純持有 {sum(r.grid_edge for r in results):+,.0f} 元"
        )
    return 0


# -------------------------------------------------------------------- fetch


def cmd_fetch(args: argparse.Namespace) -> int:
    _, portfolio, _ = _load_all(args)
    provider = _build_provider(args)
    out_dir = Path(args.out_dir)
    tickers = args.tickers or [h.ticker for h in portfolio.enabled()]
    for ticker in tickers:
        try:
            bars = provider.daily_bars(ticker, months=args.months)
        except DataError as exc:
            print(f"{ticker}：{exc}", file=sys.stderr)
            continue
        export_csv(bars, out_dir / f"{ticker}.csv")
        print(f"{ticker}：{len(bars)} 根 K 棒 → {out_dir / f'{ticker}.csv'}")
    return 0


# ---------------------------------------------------------------------- lot


def cmd_lot(args: argparse.Namespace) -> int:
    settings_path, portfolio_path, _ = _paths(args)
    try:
        settings = load_settings(settings_path)
        discount = settings.fee_discount
        minimum = settings.fee_minimum
    except ConfigError:
        discount, minimum = args.discount, 1

    if args.price:
        prices = [(f"@{p}", p, args.asset_class) for p in args.price]
    else:
        portfolio = load_portfolio(portfolio_path)
        prices = [(h.name, h.avg_cost, h.asset_class) for h in portfolio.enabled()]

    print(f"手續費折數 {float(discount):.2f}　最低 {minimum} 元\n")
    header = f"{'標的':<20}{'價格':>9}{'一份股數':>10}{'金額':>10}{'費':>5}{'來回成本%':>11}"
    print(header)
    print("-" * len(header))
    for name, price, asset_class in prices:
        shares = max_shares_for_min_fee(price, discount, minimum)
        amount = shares * price
        cost = float(round_trip_cost_pct(price, asset_class, discount, minimum))
        fee = buy_cost(shares, price, discount, minimum).fee
        print(
            f"{name:<20}{price:>9.2f}{shares:>10,}{amount:>10,.0f}{fee:>5}{cost:>11.3f}"
        )
    return 0


# --------------------------------------------------------------------- 參數


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atrgrid", description="ATR 自適應網格 · 台股零股每日建議系統"
    )
    parser.add_argument("--config-dir", help="設定檔目錄（預設 config/）")
    parser.add_argument("--state", help="狀態檔路徑（預設 state/state.json）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--provider", choices=["twse", "csv"], default="twse", help="行情來源"
        )
        p.add_argument("--csv-dir", help="csv provider 的資料夾")
        p.add_argument(
            "--price",
            action="append",
            metavar="TICKER=PRICE",
            help="覆寫即時價（可重複），搭配 csv provider 做情境模擬",
        )
        p.add_argument("--months", type=int, default=14, help="往前抓幾個月的日 K")

    p_init = sub.add_parser("init", help="建立初始網格狀態")
    add_data_args(p_init)
    p_init.add_argument("--cash", type=float, help="現金池，預設取 settings.yaml")
    p_init.add_argument("--force", action="store_true", help="覆蓋既有狀態檔")
    p_init.add_argument(
        "--fallback-to-cost",
        action="store_true",
        help="抓不到即時價時改用平均成本建檔",
    )
    p_init.set_defaults(func=cmd_init)

    p_verify = sub.add_parser("verify-tickers", help="核對股票代號")
    add_data_args(p_verify)
    p_verify.set_defaults(func=cmd_verify_tickers)

    p_advise = sub.add_parser("advise", help="產生今日建議")
    add_data_args(p_advise)
    p_advise.add_argument("--date", help="指定日期（預設今天）")
    p_advise.add_argument(
        "--format", choices=["text", "markdown", "html"], default="text"
    )
    p_advise.add_argument("--out", help="輸出檔案路徑")
    p_advise.add_argument("--json", help="同時輸出 JSON 到指定路徑")
    p_advise.add_argument(
        "--commit", action="store_true", help="把建議視為已成交寫入狀態（模擬用）"
    )
    p_advise.add_argument(
        "--persist-anchors",
        action="store_true",
        help="即使沒下單也保存錨點漂移與除息調整",
    )
    p_advise.set_defaults(func=cmd_advise)

    p_record = sub.add_parser("record", help="記錄實際成交")
    p_record.add_argument("ticker")
    p_record.add_argument("action", choices=[BUY, SELL])
    p_record.add_argument("shares", type=int)
    p_record.add_argument("price", type=float)
    p_record.add_argument("--rungs", type=int, help="這筆算幾份（預設由股數推算）")
    p_record.add_argument("--step", type=float, help="當時步長，用來精準移動錨點")
    p_record.add_argument("--date", help="成交日（預設今天）")
    p_record.add_argument("--note")
    p_record.set_defaults(func=cmd_record)

    p_status = sub.add_parser("status", help="顯示網格狀態")
    p_status.add_argument("--trades", type=int, default=10, help="顯示最近幾筆成交")
    p_status.set_defaults(func=cmd_status)

    p_bt = sub.add_parser("backtest", help="回測")
    add_data_args(p_bt)
    p_bt.add_argument("tickers", nargs="*", help="標的代號，留空代表全部")
    p_bt.add_argument("--cash", type=float, help="回測起始現金")
    p_bt.add_argument("--sweep", action="store_true", help="掃描不同 ATR 倍數")
    p_bt.add_argument("--trades", type=int, default=0, help="列出最近幾筆成交")
    p_bt.set_defaults(func=cmd_backtest)

    p_fetch = sub.add_parser("fetch", help="下載日 K 為 CSV")
    add_data_args(p_fetch)
    p_fetch.add_argument("tickers", nargs="*")
    p_fetch.add_argument("--out-dir", default="data/bars")
    p_fetch.set_defaults(func=cmd_fetch)

    p_lot = sub.add_parser("lot", help="計算「一份」是幾股")
    p_lot.add_argument("price", nargs="*", type=float, help="價格，留空則列出持股")
    p_lot.add_argument("--discount", default="0.28")
    p_lot.add_argument(
        "--asset-class", default="equity", choices=["equity", "bond", "leveraged"]
    )
    p_lot.set_defaults(func=cmd_lot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DataError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"找不到：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
