"""技術指標：True Range / Wilder ATR / EMA。

刻意只用標準函式庫，方便在任何環境（含 GitHub Actions 的乾淨容器）直接跑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Bar:
    """單日 K 棒。"""

    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"{self.date}: high {self.high} < low {self.low}")


def true_range(bar: Bar, prev_close: float | None) -> float:
    """真實區間。第一根沒有前收時退化為當日高低差。"""
    if prev_close is None:
        return bar.high - bar.low
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close),
    )


def true_ranges(bars: Sequence[Bar]) -> list[float]:
    out: list[float] = []
    prev_close: float | None = None
    for bar in bars:
        out.append(true_range(bar, prev_close))
        prev_close = bar.close
    return out


def wilder_atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Wilder 平滑 ATR。

    首值取前 ``period`` 根 TR 的簡單平均，其後
    ``ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period``。

    資料不足 ``period + 1`` 根時回傳 ``None`` ── 寧可沒有訊號，也不要用
    半成品的 ATR 去決定下單股數。
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if len(bars) < period + 1:
        return None

    trs = true_ranges(bars)[1:]  # 丟掉第一根（沒有前收，TR 失真）
    if len(trs) < period:
        return None

    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def atr_series(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """逐日 ATR，與 ``bars`` 等長；尚未成形處為 ``None``。回測用。"""
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out

    trs = true_ranges(bars)
    atr = sum(trs[1 : period + 1]) / period
    out[period] = atr
    for i in range(period + 1, len(bars)):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i] = atr
    return out


def ema(values: Sequence[float], period: int) -> float | None:
    """指數移動平均，種子值為前 ``period`` 筆的簡單平均。"""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = value * alpha + result * (1 - alpha)
    return result


def ema_series(values: Sequence[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = values[i] * alpha + current * (1 - alpha)
        out[i] = current
    return out
