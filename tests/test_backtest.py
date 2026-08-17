"""回測層級的測試：確認網格在不同市況下的行為符合設計意圖。"""

import math
import random
from dataclasses import replace

import pytest

from atrgrid.backtest import run_backtest, sweep_multiplier
from atrgrid.config import Settings
from atrgrid.indicators import Bar


def synth_bars(
    n: int,
    start: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.015,
    seed: int = 7,
    wave: float = 0.0,
    wave_period: int = 40,
) -> list[Bar]:
    """產生合成日 K。

    ``drift`` 每日漂移、``vol`` 每日波動、``wave`` 疊加正弦波（模擬區間震盪）。
    """
    from datetime import date, timedelta

    rng = random.Random(seed)
    day = date(2026, 1, 1)
    price = start
    bars = []
    for i in range(n):
        shock = rng.gauss(drift, vol)
        price *= 1 + shock
        cyclical = 1 + wave * math.sin(2 * math.pi * i / wave_period)
        close = price * cyclical
        intraday = abs(rng.gauss(0, vol)) * close + 0.005 * close
        bars.append(
            Bar(
                date=(day + timedelta(days=i)).isoformat(),
                open=close,
                high=close + intraday / 2,
                low=max(0.01, close - intraday / 2),
                close=close,
            )
        )
    return bars


@pytest.fixture
def bt_settings(settings):
    """回測用參數：步長下限放寬，讓 ATR 真正主導。"""
    params = replace(
        settings.params_for("equity"),
        atr_multiplier=0.8,
        min_step_pct=1.0,
        max_rungs_per_day=2,
    )
    return Settings(cash=500_000.0, defaults={"equity": params, "bond": params,
                                              "leveraged": params})


def test_backtest_runs_and_reports(holding, bt_settings):
    bars = synth_bars(400, wave=0.06)
    result = run_backtest(holding, bars, bt_settings)
    assert result.bars > 300
    assert result.total_trades > 0
    assert result.cash_end != result.cash_start
    assert "測試 ETF" in result.summary()


def test_range_bound_market_generates_round_trips(holding, bt_settings):
    """區間震盪是網格的主場：買賣應該成雙成對地發生。"""
    bars = synth_bars(500, drift=0.0, vol=0.008, wave=0.08, wave_period=50)
    result = run_backtest(holding, bars, bt_settings)
    assert result.buys > 3
    assert result.sells > 3
    assert result.realized_pnl > 0


def test_grid_beats_holding_in_a_choppy_market(holding, bt_settings):
    """震盪盤中，網格的低買高賣應該勝過純持有。"""
    bars = synth_bars(600, drift=0.0, vol=0.006, wave=0.10, wave_period=60, seed=11)
    result = run_backtest(holding, bars, bt_settings)
    assert result.grid_edge > 0


def test_position_limits_cap_the_downside_accumulation(holding, bt_settings):
    """一路下跌時，加碼必須停在上限，不能無限接刀。"""
    bars = synth_bars(400, drift=-0.004, vol=0.01, seed=3)
    result = run_backtest(holding, bars, bt_settings)
    max_buy = bt_settings.params_for("equity").max_buy_rungs
    assert result.max_rung <= max_buy
    assert result.days_at_buy_limit > 0  # 確實撞到上限並停手


def test_position_floor_caps_selling_in_a_bull_market(holding, bt_settings):
    bars = synth_bars(400, drift=0.004, vol=0.01, seed=5)
    result = run_backtest(holding, bars, bt_settings)
    min_sell = bt_settings.params_for("equity").max_sell_rungs
    assert result.min_rung >= -min_sell
    assert result.shares_end > 0  # 不會被賣光


def test_no_loss_sell_keeps_realized_pnl_non_negative(holding, bt_settings):
    """allow_loss_sell=false 之下，每一筆賣出都不該是虧的。"""
    bars = synth_bars(500, drift=-0.001, vol=0.012, wave=0.05, seed=13)
    result = run_backtest(holding, bars, bt_settings)
    for trade in result.trades:
        if trade["action"] == "SELL":
            assert trade["realized_pnl"] > 0


def test_wider_step_trades_less(holding, bt_settings):
    bars = synth_bars(500, wave=0.08, seed=17)
    results = dict(
        sweep_multiplier(holding, bars, bt_settings, multipliers=(0.5, 2.0))
    )
    assert results[2.0].total_trades < results[0.5].total_trades


def test_tighter_step_costs_more_in_fees(holding, bt_settings):
    bars = synth_bars(500, wave=0.08, seed=19)
    results = dict(
        sweep_multiplier(holding, bars, bt_settings, multipliers=(0.5, 2.0))
    )
    tight = results[0.5]
    wide = results[2.0]
    assert tight.total_fees + tight.total_tax >= wide.total_fees + wide.total_tax


def test_backtest_rejects_insufficient_history(holding, bt_settings):
    with pytest.raises(ValueError, match="需要至少"):
        run_backtest(holding, synth_bars(20), bt_settings)


def test_cash_never_goes_negative(holding, bt_settings):
    bars = synth_bars(500, drift=-0.003, vol=0.015, seed=23)
    small_cash = Settings(cash=20_000.0, defaults=bt_settings.defaults)
    result = run_backtest(holding, bars, small_cash, cash=20_000.0)
    assert result.cash_end >= 0


def test_bond_class_trades_without_tax(holding, bt_settings):
    bond = replace(holding, asset_class="bond")
    bars = synth_bars(500, vol=0.004, wave=0.03, seed=29)
    result = run_backtest(bond, bars, bt_settings)
    assert result.total_tax == 0
    assert result.total_fees > 0
