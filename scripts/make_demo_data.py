#!/usr/bin/env python3
"""產生示範用的合成日 K，讓整套流程可以在沒有網路的情況下跑完。

★ 這是**假資料** ★ ── 只用來驗證系統邏輯與體驗報表輸出。
正式使用請改用真實行情：

    atrgrid fetch --out-dir data/bars      # 從證交所下載
    atrgrid advise --provider twse         # 或直接連線取價

每檔標的的波動度依資產類別設定成接近現實的水準，終點價格對齊使用者
2026-08-17 的對帳市價。
"""

from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atrgrid.config import load_portfolio  # noqa: E402
from atrgrid.data import export_csv  # noqa: E402
from atrgrid.indicators import Bar  # noqa: E402

# 各資產類別的每日波動度（標準差）與盤中振幅，取近似真實的量級
VOL_BY_CLASS = {
    "equity": (0.014, 0.016),
    "bond": (0.0035, 0.0040),
    "leveraged": (0.028, 0.032),
}

# 對帳單上的市價（作為序列終點）
LAST_PRICE = {
    "0052": 61.80,
    "00685L": 11.84,
    "00725B": 34.03,
    "00735": 104.05,
    "00757": 140.30,
    "00878": 33.75,
    "00910": 63.95,
    "00933B": 15.75,
    "00937B": 14.41,
    "00947": 37.74,
    "00951": 19.85,
    "00955": 15.33,
    "TBD1": 27.26,
    "TBD2": 16.29,
    "00981A": 30.39,
    "TBD3": 10.45,
    "TBD4": 17.69,
    "TBD5": 16.72,
    "TBD6": 11.85,
}


def generate(
    ticker: str, last_price: float, asset_class: str, days: int, end: date
) -> list[Bar]:
    """反向生成序列：先隨機游走，再整體縮放讓最後一根落在 last_price。"""
    daily_vol, intraday_vol = VOL_BY_CLASS[asset_class]
    rng = random.Random(hash(ticker) & 0xFFFF)

    # 疊一點週期性，讓網格有東西可做（真實市場也不是純隨機漂移）
    closes: list[float] = []
    level = 1.0
    for i in range(days):
        level *= 1 + rng.gauss(0, daily_vol)
        cycle = 1 + 0.04 * math.sin(2 * math.pi * i / 55)
        closes.append(level * cycle)

    scale = last_price / closes[-1]
    closes = [c * scale for c in closes]

    bars: list[Bar] = []
    day = end - timedelta(days=days - 1)
    for i, close in enumerate(closes):
        span = abs(rng.gauss(0, intraday_vol)) * close + 0.004 * close
        high = close + span * rng.uniform(0.3, 0.7)
        low = max(0.01, close - span * rng.uniform(0.3, 0.7))
        open_ = low + (high - low) * rng.random()
        bars.append(
            Bar(
                date=(day + timedelta(days=i)).isoformat(),
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(rng.uniform(1e6, 2e7)),
            )
        )
    return bars


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    portfolio = load_portfolio(root / "config" / "portfolio.yaml")
    out_dir = root / "data" / "bars"

    # 最後一根收在昨天，模擬「今天盤中尚未收盤」
    end = date.today() - timedelta(days=1)
    days = 420

    for holding in portfolio.holdings:
        price = LAST_PRICE.get(holding.ticker, holding.avg_cost)
        bars = generate(holding.ticker, price, holding.asset_class, days, end)
        export_csv(bars, out_dir / f"{holding.ticker}.csv")
        print(
            f"{holding.ticker:<8}{holding.name:<18}{len(bars):>5} 根　"
            f"收於 {bars[-1].close:>8.2f}（{bars[-1].date}）"
        )

    print(f"\n示範資料已寫入 {out_dir}（合成資料，非真實行情）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
