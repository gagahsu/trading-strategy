"""把決策整理成可讀的報表（終端機 / Markdown / HTML）。"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date

from .engine import BUY, HOLD, REVIEW, SELL, SKIP, Decision
from .state import State

ACTION_LABEL = {
    BUY: "買進",
    SELL: "賣出",
    HOLD: "觀望",
    REVIEW: "人工複核",
    SKIP: "略過",
}

ACTION_ORDER = {BUY: 0, SELL: 1, REVIEW: 2, HOLD: 3, SKIP: 4}


def order_spec(decision: Decision) -> str:
    """把決策表示成「幾筆 × 每筆幾股」。

    每一「份」都是獨立一筆委託 —— 合併下單會突破手續費 1 元的門檻。
    """
    if not decision.shares:
        return "—"
    return f"{decision.rungs} × {decision.lot_shares:,}"


@dataclass
class ReportContext:
    as_of: str
    decision_time: str
    state: State
    decisions: list[Decision]
    warnings: list[str]

    def actionable(self) -> list[Decision]:
        return [d for d in self.decisions if d.is_actionable]

    def sorted_decisions(self) -> list[Decision]:
        return sorted(
            self.decisions,
            key=lambda d: (ACTION_ORDER.get(d.action, 9), -abs(d.est_cash_flow)),
        )

    def net_cash_flow(self) -> float:
        return sum(d.est_cash_flow for d in self.actionable())

    def total_fees(self) -> int:
        return sum(d.est_fee + d.est_tax for d in self.actionable())


def render_text(ctx: ReportContext) -> str:
    """終端機輸出。"""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  ATR 網格 · 每日建議   {ctx.as_of} {ctx.decision_time}")
    lines.append("=" * 78)

    for warning in ctx.warnings:
        lines.append(f"  ⚠  {warning}")
    if ctx.warnings:
        lines.append("")

    actionable = ctx.actionable()
    if not actionable:
        lines.append("  今日無建議下單（所有標的皆未觸及網格）。")
    else:
        lines.append(f"  今日建議 {len(actionable)} 筆：")
        lines.append("")
        header = (
            f"  {'動作':<6}{'標的':<18}{'下單':>12}{'總股數':>8}"
            f"{'價格':>9}{'金額':>10}{'費+稅':>7}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for d in actionable:
            lines.append(
                f"  {ACTION_LABEL[d.action]:<6}{d.name[:16]:<18}"
                f"{order_spec(d):>12}{d.shares:>8,}{d.price:>9.2f}"
                f"{d.est_gross:>10,.0f}{d.est_fee + d.est_tax:>7}"
            )
        lines.append("")
        lines.append("  ※「下單」欄的 N × M 代表分 N 筆、每筆 M 股 —— 拆單才能維持每筆 1 元手續費")
        lines.append("")
        lines.append(
            f"  淨現金流 {ctx.net_cash_flow():>+12,.0f} 元　"
            f"交易成本 {ctx.total_fees():,} 元　"
            f"餘額 {ctx.state.cash + ctx.net_cash_flow():,.0f} 元"
        )

    lines.append("")
    lines.append("-" * 78)
    lines.append("  全部標的")
    lines.append("-" * 78)
    for d in ctx.sorted_decisions():
        head = (
            f"  [{ACTION_LABEL[d.action]}] {d.name} ({d.ticker})　"
            f"現價 {d.price:.2f}　錨點 {d.anchor_before:.2f}"
        )
        if d.step:
            head += f"　步長 {d.step:.3f}（{d.step_pct:.2f}%）"
        lines.append(head)
        for reason in d.reasons:
            lines.append(f"       · {reason}")
        for note in d.notes:
            lines.append(f"       ~ {note}")
        for block in d.blocks:
            lines.append(f"       ✗ {block}")
        if d.action == SELL and d.est_realized_pnl is not None:
            lines.append(f"       $ 預估實現損益 {d.est_realized_pnl:+,.0f} 元")
    lines.append("")
    return "\n".join(lines)


def render_markdown(ctx: ReportContext) -> str:
    lines: list[str] = []
    lines.append(f"# ATR 網格每日建議 · {ctx.as_of} {ctx.decision_time}")
    lines.append("")

    if ctx.warnings:
        lines.append("> [!WARNING]")
        for warning in ctx.warnings:
            lines.append(f"> - {warning}")
        lines.append("")

    actionable = ctx.actionable()
    lines.append("## 今日下單")
    lines.append("")
    if not actionable:
        lines.append("今日無建議下單 — 所有標的都還在格子內。")
    else:
        lines.append(
            "| 動作 | 標的 | 代號 | 下單（筆×股） | 總股數 | 參考價 | 金額 "
            "| 費+稅 | 現金流 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for d in actionable:
            lines.append(
                f"| **{ACTION_LABEL[d.action]}** | {d.name} | `{d.ticker}` | "
                f"{order_spec(d)} | {d.shares:,} | {d.price:.2f} | "
                f"{d.est_gross:,.0f} | {d.est_fee + d.est_tax} | "
                f"{d.est_cash_flow:+,.0f} |"
            )
        lines.append("")
        lines.append(
            "「下單」欄的 N × M 代表**分 N 筆、每筆 M 股**送出 —— "
            "一份是手續費 1 元的最大股數，合併成一筆會多付費用。"
        )
        lines.append("")
        lines.append(
            f"淨現金流 **{ctx.net_cash_flow():+,.0f}** 元 · "
            f"交易成本 **{ctx.total_fees():,}** 元 · "
            f"預估餘額 **{ctx.state.cash + ctx.net_cash_flow():,.0f}** 元"
        )
    lines.append("")

    lines.append("## 全部標的")
    lines.append("")
    lines.append(
        "| 狀態 | 標的 | 現價 | 錨點 | ATR% | 步長% | 一份 | 階數 | 持股 | 說明 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for d in ctx.sorted_decisions():
        detail = "；".join(d.reasons + d.notes + [f"✗ {b}" for b in d.blocks]) or "—"
        atr_pct = f"{d.atr_pct:.2f}" if d.atr_pct is not None else "—"
        step_pct = f"{d.step_pct:.2f}" if d.step_pct else "—"
        lines.append(
            f"| {ACTION_LABEL[d.action]} | {d.name} `{d.ticker}` | {d.price:.2f} | "
            f"{d.anchor_before:.2f} | {atr_pct} | {step_pct} | {d.lot_shares or '—'} | "
            f"{d.rung_before:+d} | {d.position_shares:,} | {detail} |"
        )
    lines.append("")
    lines.append(
        "> 本報表由 ATR 網格系統自動產生，僅為依既定規則計算的機械式提示，"
        "不構成投資建議。下單前請自行確認價格、除息與市場狀況。"
    )
    lines.append("")
    return "\n".join(lines)


def render_html(ctx: ReportContext) -> str:
    """自成一體的 HTML 報表，深淺色皆可讀。"""

    def esc(value: object) -> str:
        return html.escape(str(value))

    badge = {
        BUY: ("買進", "buy"),
        SELL: ("賣出", "sell"),
        HOLD: ("觀望", "hold"),
        REVIEW: ("複核", "review"),
        SKIP: ("略過", "skip"),
    }

    rows = []
    for d in ctx.sorted_decisions():
        label, cls = badge.get(d.action, ("—", "skip"))
        detail = "；".join(d.reasons + d.notes + [f"✗ {b}" for b in d.blocks]) or "—"
        atr_pct = f"{d.atr_pct:.2f}%" if d.atr_pct is not None else "—"
        step_pct = f"{d.step_pct:.2f}%" if d.step_pct else "—"
        qty = order_spec(d)
        rows.append(
            f"<tr>"
            f'<td><span class="badge {cls}">{label}</span></td>'
            f'<td class="name">{esc(d.name)}<span class="tk">{esc(d.ticker)}</span></td>'
            f'<td class="n">{d.price:.2f}</td>'
            f'<td class="n">{d.anchor_before:.2f}</td>'
            f'<td class="n">{atr_pct}</td>'
            f'<td class="n">{step_pct}</td>'
            f'<td class="n">{qty}</td>'
            f'<td class="n">{d.rung_before:+d}</td>'
            f'<td class="n">{d.position_shares:,}</td>'
            f'<td class="detail">{esc(detail)}</td>'
            f"</tr>"
        )

    actionable = ctx.actionable()
    summary_cards = [
        ("委託筆數", f"{sum(d.rungs for d in actionable)}", "筆"),
        ("淨現金流", f"{ctx.net_cash_flow():+,.0f}", "元"),
        ("交易成本", f"{ctx.total_fees():,}", "元"),
        ("現金餘額", f"{ctx.state.cash + ctx.net_cash_flow():,.0f}", "元"),
    ]
    cards = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div>'
        f'<div class="v">{esc(v)}<span class="u">{esc(u)}</span></div></div>'
        for k, v, u in summary_cards
    )

    warnings = ""
    if ctx.warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in ctx.warnings)
        warnings = f'<div class="warn"><ul>{items}</ul></div>'

    return f"""<title>ATR 網格每日建議</title>
<style>
:root {{
  --bg:#f7f7f5; --panel:#fff; --ink:#1a1a18; --muted:#6b6b66; --line:#e3e3de;
  --buy:#0f7b6c; --buy-bg:#dcf2ee; --sell:#a8410f; --sell-bg:#fbe8dd;
  --hold:#5a5a54; --hold-bg:#ececea; --review:#8a6d0b; --review-bg:#faf0cf;
  --accent:#c8622a;
}}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#171714; --panel:#201f1c; --ink:#efeee9; --muted:#a09e97; --line:#33322d;
    --buy:#5fd3bd; --buy-bg:#123832; --sell:#f0a173; --sell-bg:#3d2213;
    --hold:#a09e97; --hold-bg:#2a2925; --review:#e3c261; --review-bg:#3a3113;
    --accent:#e2884d;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#171714; --panel:#201f1c; --ink:#efeee9; --muted:#a09e97; --line:#33322d;
  --buy:#5fd3bd; --buy-bg:#123832; --sell:#f0a173; --sell-bg:#3d2213;
  --hold:#a09e97; --hold-bg:#2a2925; --review:#e3c261; --review-bg:#3a3113;
  --accent:#e2884d;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",
    "PingFang TC","Microsoft JhengHei",sans-serif;
  line-height:1.55; font-size:15px;
}}
.wrap {{ max-width:1120px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); font-size:.9rem; margin-bottom:1.5rem; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:.75rem; margin-bottom:1.5rem; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:.85rem 1rem; }}
.card .k {{ color:var(--muted); font-size:.75rem; letter-spacing:.04em; }}
.card .v {{ font-size:1.5rem; font-weight:600; font-variant-numeric:tabular-nums; }}
.card .u {{ font-size:.8rem; color:var(--muted); margin-left:.2rem; font-weight:400; }}
.warn {{ background:var(--review-bg); border-left:3px solid var(--review);
  border-radius:6px; padding:.6rem 1rem; margin-bottom:1.25rem; }}
.warn ul {{ margin:0; padding-left:1.1rem; }}
h2 {{ font-size:1.05rem; margin:1.75rem 0 .6rem; }}
.scroll {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:.86rem; min-width:900px; }}
th {{ text-align:left; padding:.6rem .7rem; color:var(--muted); font-weight:500;
  border-bottom:1px solid var(--line); white-space:nowrap; font-size:.78rem;
  letter-spacing:.03em; }}
td {{ padding:.55rem .7rem; border-bottom:1px solid var(--line);
  vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.name {{ white-space:nowrap; font-weight:500; }}
td.name .tk {{ color:var(--muted); font-size:.75rem; margin-left:.4rem;
  font-weight:400; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
td.detail {{ color:var(--muted); font-size:.8rem; min-width:280px; }}
.badge {{ display:inline-block; padding:.12rem .5rem; border-radius:5px;
  font-size:.76rem; font-weight:600; white-space:nowrap; }}
.badge.buy {{ color:var(--buy); background:var(--buy-bg); }}
.badge.sell {{ color:var(--sell); background:var(--sell-bg); }}
.badge.hold {{ color:var(--hold); background:var(--hold-bg); }}
.badge.review {{ color:var(--review); background:var(--review-bg); }}
.badge.skip {{ color:var(--muted); background:var(--hold-bg); }}
footer {{ margin-top:2rem; color:var(--muted); font-size:.78rem;
  border-top:1px solid var(--line); padding-top:1rem; }}
</style>
<div class="wrap">
  <h1>ATR 網格每日建議</h1>
  <div class="sub">{esc(ctx.as_of)} {esc(ctx.decision_time)} · 台北時間</div>
  {warnings}
  <div class="cards">{cards}</div>
  <h2>全部標的</h2>
  <div class="scroll">
    <table>
      <thead><tr>
        <th>狀態</th><th>標的</th><th>現價</th><th>錨點</th><th>ATR%</th>
        <th>步長%</th><th>下單 筆×股</th><th>階數</th><th>持股</th><th>說明</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
  <footer>
    「下單」欄的 N × M 代表分 N 筆、每筆 M 股送出：一份是手續費 1 元的最大股數，
    合併成一筆會多付費用。<br>
    由 ATR 網格系統依既定規則自動計算，為機械式提示而非投資建議。
    下單前請自行確認即時價格、除權息與市場狀況。
  </footer>
</div>
"""


def render_state_summary(state: State) -> str:
    """狀態摘要，供 `atrgrid status` 使用。"""
    lines = [
        f"現金餘額　{state.cash:,.0f} 元",
        f"最後執行　{state.last_run_date or '尚未執行'}",
        f"成交筆數　{len(state.trades)}",
    ]
    realized = sum(t.realized_pnl for t in state.trades)
    lines.append(f"網格已實現損益　{realized:+,.0f} 元")
    lines.append("")
    header = f"{'標的':<10}{'持股':>9}{'階數':>6}{'錨點':>10}{'均成本':>10}{'實現損益':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for ticker, position in sorted(state.positions.items()):
        lines.append(
            f"{ticker:<10}{position.shares:>9,}{position.rung:>+6d}"
            f"{position.anchor:>10.2f}{position.average_cost():>10.2f}"
            f"{position.realized_pnl:>+12,.0f}"
        )
    return "\n".join(lines)


def today_iso() -> str:
    return date.today().isoformat()
