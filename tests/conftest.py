import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atrgrid.config import GridParams, Holding, Settings  # noqa: E402
from atrgrid.indicators import Bar  # noqa: E402
from atrgrid.state import Lot, Position, State  # noqa: E402

#: 測試共用的「今天」。K 棒一律以此往前生成，最後一根落在前一天，
#: 才不會被引擎的資料過期檢查擋掉。
TODAY = "2026-03-15"


def bars_with_atr(
    count: int = 60,
    close: float = 100.0,
    spread: float = 2.0,
    trend: float = 0.0,
    end_date: str = TODAY,
) -> list[Bar]:
    """產生一組高低差固定的日 K，方便讓 ATR 收斂到已知值。

    最後一根收在 ``end_date`` 的前一天，模擬「今天尚未收盤」的實際情境。
    ``close`` 是最後一根的收盤價，``trend`` 是每日漲跌幅（往回推算）。
    """
    from datetime import date as _date
    from datetime import timedelta

    last = _date.fromisoformat(end_date) - timedelta(days=1)
    bars = []
    for i in range(count):
        offset = count - 1 - i  # 距離最後一根幾天
        c = close - trend * offset
        bars.append(
            Bar(
                date=(last - timedelta(days=offset)).isoformat(),
                open=c,
                high=c + spread / 2,
                low=c - spread / 2,
                close=c,
            )
        )
    return bars


@pytest.fixture
def settings() -> Settings:
    base = GridParams(
        atr_period=14,
        atr_multiplier=0.5,
        min_step_pct=0.1,
        max_step_pct=20.0,
        max_buy_rungs=5,
        max_sell_rungs=5,
        max_rungs_per_day=2,
        gap_atr_limit=3.0,
        drift_mode="off",
        drift_beta=0.0,
        trend_ema_period=60,
        allow_loss_sell=False,
    )
    return Settings(
        cash=1_000_000.0,
        defaults={
            "equity": base,
            "bond": base,
            "leveraged": base,
        },
    )


@pytest.fixture
def holding() -> Holding:
    return Holding(
        ticker="0052",
        name="測試 ETF",
        asset_class="equity",
        shares=1000,
        avg_cost=50.0,
        ticker_verified=True,
    )


@pytest.fixture
def position() -> Position:
    return Position(
        ticker="0052",
        shares=1000,
        anchor=100.0,
        rung=0,
        baseline_shares=1000,
        lots=[Lot(date="2026-01-01", price=50.0, shares=1000, source="initial")],
    )


@pytest.fixture
def state(position) -> State:
    return State(cash=1_000_000.0, positions={"0052": position})
