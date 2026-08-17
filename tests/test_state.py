import json

import pytest

from atrgrid.state import Lot, Position, State, load_state, save_state


def make_position() -> Position:
    return Position(
        ticker="0052",
        shares=300,
        anchor=100.0,
        baseline_shares=300,
        lots=[
            Lot(date="2026-01-01", price=50.0, shares=100, source="initial"),
            Lot(date="2026-02-01", price=90.0, shares=100),
            Lot(date="2026-03-01", price=95.0, shares=100),
        ],
    )


def test_average_cost():
    position = make_position()
    assert position.average_cost() == pytest.approx((50 + 90 + 95) * 100 / 300)


def test_sell_matches_most_recent_lot_first():
    """後進先出：先賣掉最近一筆 95 元的批次。"""
    position = make_position()
    pnl = position.apply_sell("2026-03-10", price=100.0, shares=100, rungs=1)
    assert pnl == pytest.approx((100 - 95) * 100)
    assert position.shares == 200
    assert position.rung == -1
    assert len(position.lots) == 2
    assert position.lots[-1].price == 90.0


def test_sell_spanning_multiple_lots():
    position = make_position()
    pnl = position.apply_sell("2026-03-10", price=100.0, shares=150, rungs=1)
    # 95 元那批 100 股 + 90 元那批 50 股
    expected = (100 - 95) * 100 + (100 - 90) * 50
    assert pnl == pytest.approx(expected)
    assert position.lots[-1].shares == 50


def test_peek_sell_basis_does_not_mutate():
    position = make_position()
    before = [(lot.price, lot.shares) for lot in position.lots]
    basis = position.peek_sell_basis(150)
    assert basis == pytest.approx(95 * 100 + 90 * 50)
    assert [(lot.price, lot.shares) for lot in position.lots] == before


def test_sell_more_than_recorded_lots_falls_back_to_average():
    """狀態與券商實際庫存不同步時不該炸掉。"""
    position = Position(ticker="X", shares=100, anchor=10.0, lots=[])
    pnl = position.apply_sell("2026-03-10", price=10.0, shares=100, rungs=1)
    assert pnl == pytest.approx(0.0)
    assert position.shares == 0


def test_buy_appends_a_lot():
    position = make_position()
    position.apply_buy("2026-03-11", price=88.0, shares=50, rungs=1)
    assert position.shares == 350
    assert position.rung == 1
    assert position.lots[-1].price == 88.0


def test_round_trip_persistence(tmp_path):
    state = State(
        cash=123456.78,
        positions={"0052": make_position()},
        last_run_date="2026-03-14",
    )
    path = tmp_path / "state.json"
    save_state(state, path)
    restored = load_state(path)

    assert restored.cash == pytest.approx(123456.78)
    assert restored.last_run_date == "2026-03-14"
    position = restored.positions["0052"]
    assert position.shares == 300
    assert len(position.lots) == 3
    assert position.lots[0].source == "initial"


def test_save_creates_backup(tmp_path):
    path = tmp_path / "state.json"
    save_state(State(cash=1.0), path)
    save_state(State(cash=2.0), path)
    assert path.with_suffix(".json.bak").exists()
    assert load_state(path).cash == pytest.approx(2.0)


def test_missing_state_file_returns_empty_state(tmp_path):
    state = load_state(tmp_path / "nope.json")
    assert state.positions == {}
    assert state.cash == 0.0


def test_future_state_version_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "positions": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="版本"):
        load_state(path)


def test_saved_json_is_human_readable(tmp_path):
    path = tmp_path / "state.json"
    save_state(State(cash=1.0, positions={"0052": make_position()}), path)
    text = path.read_text(encoding="utf-8")
    assert "0052" in text
    assert "\n" in text  # 有縮排，可以直接讀與手改
