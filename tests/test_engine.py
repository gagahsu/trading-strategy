from dataclasses import replace

import pytest
from conftest import TODAY, bars_with_atr

from atrgrid.config import Settings
from atrgrid.engine import (
    BUY,
    HOLD,
    REVIEW,
    SELL,
    SKIP,
    commit,
    evaluate,
    grid_step,
    lot_size,
)
from atrgrid.state import Lot


def _evaluate(holding, position, settings, state, price, bars=None, today=TODAY):
    bars = bars if bars is not None else bars_with_atr(60, close=100.0, spread=2.0)
    return evaluate(holding, position, bars, price, settings, state, today=today)


# ------------------------------------------------------------------ 基本行為


def test_lot_size_tracks_price(settings):
    assert lot_size(100.0, settings) == 50  # 50 * 100 = 5000 < 5012.53
    assert lot_size(50.0, settings) == 100


def test_grid_step_clamped_by_floor(settings):
    params = settings.params_for("equity")
    # ATR 0.2、價格 100 → 0.5 * 0.2 = 0.1，低於下限 0.1% * 100 = 0.1
    assert grid_step(100.0, 0.2, params) == pytest.approx(0.1)


def test_grid_step_clamped_by_ceiling(settings):
    params = replace(settings.params_for("equity"), max_step_pct=1.0)
    assert grid_step(100.0, 50.0, params) == pytest.approx(1.0)


def test_no_signal_inside_the_grid(holding, position, settings, state):
    # ATR ≈ 2.0 → 步長 ≈ 1.0。價格距錨點 0.5 元，不到一格。
    decision = _evaluate(holding, position, settings, state, price=100.5)
    assert decision.action == HOLD
    assert decision.shares == 0


def test_buy_signal_when_price_drops_one_step(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=98.9)
    assert decision.action == BUY
    assert decision.rungs == 1
    assert decision.shares == lot_size(98.9, settings)
    assert decision.est_cash_flow < 0
    assert decision.anchor_after < decision.anchor_before


def test_sell_signal_when_price_rises_one_step(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=101.1)
    assert decision.action == SELL
    assert decision.rungs == 1
    assert decision.est_cash_flow > 0
    assert decision.anchor_after > decision.anchor_before


def test_multiple_rungs_when_price_moves_far(holding, position, settings, state):
    # 步長約 1.0，跌 2.5 元 → 2 格（受單日上限 2 份限制）
    decision = _evaluate(holding, position, settings, state, price=97.5)
    assert decision.action == BUY
    assert decision.rungs == 2


def test_daily_rung_cap_is_enforced(holding, position, settings, state):
    tight = replace(settings.params_for("equity"), max_rungs_per_day=1)
    settings = Settings(cash=settings.cash, defaults={"equity": tight})
    decision = _evaluate(holding, position, settings, state, price=95.0)
    assert decision.rungs == 1
    assert any("單日上限" in n for n in decision.notes)


# ------------------------------------------------------------------ 風控閘門


def test_unverified_ticker_is_blocked(holding, position, settings, state):
    holding = replace(holding, ticker_verified=False)
    decision = _evaluate(holding, position, settings, state, price=90.0)
    assert decision.action == SKIP
    assert any("代號" in b for b in decision.blocks)


def test_insufficient_bars_is_skipped(holding, position, settings, state):
    decision = _evaluate(
        holding, position, settings, state, price=98.0, bars=bars_with_atr(5)
    )
    assert decision.action == SKIP
    assert any("ATR" in b for b in decision.blocks)


def test_stale_data_is_skipped(holding, position, settings, state):
    # K 棒停在 3 月中，卻在年底才執行 → 資料過期
    decision = _evaluate(
        holding, position, settings, state, price=98.0, today="2026-12-31"
    )
    assert decision.action == SKIP
    assert any("未更新" in b for b in decision.blocks)


def test_large_gap_triggers_manual_review(holding, position, settings, state):
    # ATR ≈ 2.0，跳空 10 元 = 5 倍 ATR，超過 3 倍上限
    decision = _evaluate(holding, position, settings, state, price=90.0)
    assert decision.action == REVIEW
    assert decision.shares == 110
    assert any("跳空" in r for r in decision.reasons)


def test_buy_blocked_at_position_limit(holding, position, settings, state):
    position.rung = 5  # 已達 max_buy_rungs
    decision = _evaluate(holding, position, settings, state, price=98.9)
    assert decision.shares == 0
    assert any("加碼至上限" in b for b in decision.blocks)


def test_buy_trimmed_to_remaining_room(holding, position, settings, state):
    position.rung = 4  # 只剩 1 階
    decision = _evaluate(holding, position, settings, state, price=97.5)
    assert decision.rungs == 1
    assert any("部位上限" in n for n in decision.notes)


def test_sell_blocked_at_position_floor(holding, position, settings, state):
    position.rung = -5
    decision = _evaluate(holding, position, settings, state, price=101.1)
    assert decision.shares == 0
    assert any("減碼至下限" in b for b in decision.blocks)


def test_buy_blocked_without_cash(holding, position, settings, state):
    state.cash = 100.0
    decision = _evaluate(holding, position, settings, state, price=98.9)
    assert decision.shares == 0
    assert any("現金不足" in b for b in decision.blocks)


def test_buy_trimmed_to_available_cash(holding, position, settings, state):
    state.cash = 5_500.0  # 只夠一份
    decision = _evaluate(holding, position, settings, state, price=97.5)
    assert decision.rungs == 1


def test_cash_floor_is_respected(holding, position, settings, state):
    state.cash = 10_000.0
    settings = replace(settings, cash_floor=9_000.0)
    decision = _evaluate(holding, position, settings, state, price=98.9)
    assert decision.shares == 0


def test_sell_blocked_when_holding_less_than_one_lot(holding, position, settings, state):
    position.shares = 10
    position.lots = [Lot(date="2026-01-01", price=50.0, shares=10)]
    decision = _evaluate(holding, position, settings, state, price=101.1)
    assert decision.shares == 0
    assert any("不足一份" in b for b in decision.blocks)


def test_loss_making_sell_is_blocked(holding, position, settings, state):
    # 這批成本 200，現價 101 賣出必虧
    position.lots = [Lot(date="2026-01-01", price=200.0, shares=1000)]
    decision = _evaluate(holding, position, settings, state, price=101.1)
    assert decision.shares == 0
    assert any("虧損賣出" in b for b in decision.blocks)


def test_loss_making_sell_allowed_when_configured(holding, position, settings, state):
    position.lots = [Lot(date="2026-01-01", price=200.0, shares=1000)]
    loose = replace(settings.params_for("equity"), allow_loss_sell=True)
    settings = Settings(cash=settings.cash, defaults={"equity": loose})
    decision = _evaluate(holding, position, settings, state, price=101.1)
    assert decision.action == SELL
    assert decision.shares > 0
    assert decision.est_realized_pnl < 0


# -------------------------------------------------------------- 除息與漂移


def test_ex_dividend_lowers_anchor_before_signalling(holding, position, settings, state):
    """除息 3 元不該被誤判成下跌 3 元。"""
    holding = replace(
        holding, ex_dividends=[{"date": "2026-03-10", "amount": 3.0}]
    )
    decision = _evaluate(holding, position, settings, state, price=100.0)
    assert position.anchor == pytest.approx(97.0)
    assert decision.action == SELL  # 相對除息後的錨點，價格反而偏高
    assert any("除息" in n for n in decision.notes)


def test_ex_dividend_applied_only_once(holding, position, settings, state):
    holding = replace(holding, ex_dividends=[{"date": "2026-03-10", "amount": 3.0}])
    _evaluate(holding, position, settings, state, price=100.0)
    anchor_after_first = position.anchor
    _evaluate(holding, position, settings, state, price=100.0)
    assert position.anchor == pytest.approx(anchor_after_first)


def test_future_ex_dividend_is_not_applied(holding, position, settings, state):
    holding = replace(holding, ex_dividends=[{"date": "2026-09-10", "amount": 3.0}])
    _evaluate(holding, position, settings, state, price=100.0)
    assert position.anchor == pytest.approx(100.0)


def test_up_only_drift_raises_anchor_in_uptrend(holding, position, settings, state):
    drifting = replace(
        settings.params_for("equity"),
        drift_mode="up_only",
        drift_beta=0.1,
        trend_ema_period=20,
    )
    settings = Settings(cash=settings.cash, defaults={"equity": drifting})
    bars = bars_with_atr(80, close=140.0, spread=2.0, trend=0.5)  # 明顯上升趨勢
    position.anchor = 100.0
    _evaluate(holding, position, settings, state, price=bars[-1].close, bars=bars)
    assert position.anchor > 100.0


def test_up_only_drift_never_lowers_the_anchor(holding, position, settings, state):
    """錨點高於均線時不該被往下拉 —— 否則系統會系統性地賣在低點。"""
    drifting = replace(
        settings.params_for("equity"),
        drift_mode="up_only",
        drift_beta=0.1,
        trend_ema_period=20,
    )
    settings = Settings(cash=settings.cash, defaults={"equity": drifting})
    bars = bars_with_atr(80, close=100.0, spread=2.0, trend=-0.5)  # 下降趨勢
    position.anchor = 130.0  # 明顯高於 EMA20
    _evaluate(holding, position, settings, state, price=bars[-1].close, bars=bars)
    assert position.anchor == pytest.approx(130.0)


def test_both_drift_mode_does_lower_the_anchor(holding, position, settings, state):
    two_way = replace(
        settings.params_for("equity"),
        drift_mode="both",
        drift_beta=0.1,
        trend_ema_period=20,
    )
    settings = Settings(cash=settings.cash, defaults={"equity": two_way})
    bars = bars_with_atr(80, close=100.0, spread=2.0, trend=-0.5)
    position.anchor = 130.0
    _evaluate(holding, position, settings, state, price=bars[-1].close, bars=bars)
    assert position.anchor < 130.0


# ------------------------------------------------------------------- commit


def test_commit_buy_updates_everything(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=98.9)
    cash_before = state.cash
    commit(state, decision, trade_date=TODAY)

    assert position.shares == 1000 + decision.shares
    assert position.rung == 1
    assert position.anchor == pytest.approx(decision.anchor_after)
    assert state.cash == pytest.approx(cash_before + decision.est_cash_flow)
    assert len(state.trades) == 1
    assert state.trades[0].action == BUY


def test_commit_sell_realizes_profit(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=101.1)
    commit(state, decision, trade_date=TODAY)

    assert position.shares == 1000 - decision.shares
    assert position.rung == -1
    assert state.trades[0].realized_pnl > 0  # 成本 50，賣在 101
    assert state.cash > 1_000_000.0


def test_round_trip_buy_then_sell_returns_to_origin(holding, position, settings, state):
    buy = _evaluate(holding, position, settings, state, price=98.9)
    commit(state, buy, trade_date=TODAY)
    assert position.rung == 1

    sell = _evaluate(holding, position, settings, state, price=100.2, today="2026-03-16")
    assert sell.action == SELL
    commit(state, sell, trade_date="2026-03-16")
    assert position.rung == 0
    # 低買高賣，這一趟該是賺的
    assert state.trades[-1].realized_pnl > 0


def test_hold_decision_is_not_committed(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=100.2)
    commit(state, decision, trade_date=TODAY)
    assert state.trades == []
    assert position.shares == 1000


def test_multi_rung_order_is_costed_as_separate_orders(
    holding, position, settings, state
):
    """買 2 份的手續費應該是 1+1，而不是合併下單的 3 元。"""
    decision = _evaluate(holding, position, settings, state, price=97.5)
    assert decision.rungs == 2
    assert decision.est_fee == 2
    assert any("分 2 筆" in n for n in decision.notes)


def test_single_rung_has_no_split_note(holding, position, settings, state):
    decision = _evaluate(holding, position, settings, state, price=98.9)
    assert decision.est_fee == 1
    assert not any("分" in n and "筆" in n for n in decision.notes)


def test_drift_applies_only_once_per_day(holding, position, settings, state):
    """同一天重複執行 advise 不該把錨點越推越遠。"""
    drifting = replace(
        settings.params_for("equity"),
        drift_mode="up_only",
        drift_beta=0.1,
        trend_ema_period=20,
    )
    settings = Settings(cash=settings.cash, defaults={"equity": drifting})
    bars = bars_with_atr(80, close=140.0, spread=2.0, trend=0.5)

    position.anchor = 100.0
    _evaluate(holding, position, settings, state, price=140.0, bars=bars)
    after_first = position.anchor
    assert after_first > 100.0

    _evaluate(holding, position, settings, state, price=140.0, bars=bars)
    assert position.anchor == pytest.approx(after_first)


def test_drift_resumes_the_next_day(holding, position, settings, state):
    drifting = replace(
        settings.params_for("equity"),
        drift_mode="up_only",
        drift_beta=0.1,
        trend_ema_period=20,
    )
    settings = Settings(cash=settings.cash, defaults={"equity": drifting})
    bars = bars_with_atr(80, close=140.0, spread=2.0, trend=0.5)

    position.anchor = 100.0
    _evaluate(holding, position, settings, state, price=140.0, bars=bars)
    after_first = position.anchor
    _evaluate(
        holding, position, settings, state, price=140.0, bars=bars, today="2026-03-16"
    )
    assert position.anchor > after_first
