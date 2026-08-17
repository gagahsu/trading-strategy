from decimal import Decimal

import pytest

from atrgrid.fees import (
    brokerage_fee,
    buy_cost,
    max_shares_for_min_fee,
    round_trip_cost_pct,
    sell_cost,
    transaction_tax,
)


def test_fee_is_floored_not_rounded():
    # 10000 * 0.001425 * 0.28 = 3.99 → 捨去為 3
    assert brokerage_fee(10000) == 3


def test_minimum_fee_applies():
    assert brokerage_fee(100) == 1
    assert brokerage_fee(1) == 1


def test_zero_amount_costs_nothing():
    assert brokerage_fee(0) == 0


@pytest.mark.parametrize(
    "price,expected",
    [
        (61.80, 81),
        (11.84, 423),
        (34.03, 147),
        (140.30, 35),
        (10.45, 479),
    ],
)
def test_lot_size_matches_hand_calculation(price, expected):
    assert max_shares_for_min_fee(price) == expected


@pytest.mark.parametrize(
    "price",
    [5.0, 10.45, 11.84, 15.75, 27.26, 34.03, 61.8, 104.05, 140.3, 500.0],
)
def test_lot_size_is_exactly_at_the_boundary(price):
    """一份的手續費必須是 1 元，而多買 1 股必須跳到 2 元。"""
    shares = max_shares_for_min_fee(price)
    assert brokerage_fee(Decimal(str(price)) * shares) == 1
    assert brokerage_fee(Decimal(str(price)) * (shares + 1)) == 2


def test_lot_size_never_below_one_share():
    # 超高價標的：1 股就超過門檻，仍應回傳 1
    assert max_shares_for_min_fee(100000) == 1


def test_lot_size_respects_other_discounts():
    """換券商折數，臨界股數要跟著變。6 折的門檻金額比 2.8 折低。"""
    at_28 = max_shares_for_min_fee(50, discount="0.28")
    at_60 = max_shares_for_min_fee(50, discount="0.60")
    assert at_60 < at_28
    assert brokerage_fee(Decimal("50") * at_60, discount="0.60") == 1


def test_bond_etf_is_tax_exempt():
    assert transaction_tax(100000, "bond") == 0
    assert transaction_tax(100000, "equity") == 100


def test_tax_is_floored():
    # 5005 * 0.001 = 5.005 → 5
    assert transaction_tax(5005, "equity") == 5


def test_round_trip_cost_bond_is_cheaper_than_equity():
    bond = float(round_trip_cost_pct(34.03, "bond"))
    equity = float(round_trip_cost_pct(34.03, "equity"))
    assert bond < equity
    assert bond == pytest.approx(0.04, abs=0.005)
    assert equity == pytest.approx(0.14, abs=0.03)


def test_buy_and_sell_cash_flows():
    buy = buy_cost(81, 61.80)
    assert buy.fee == 1
    assert float(buy.net) == pytest.approx(5006.8)

    sell = sell_cost(81, 61.80, "equity")
    assert sell.fee == 1
    assert sell.tax == 5  # 5005.8 * 0.001 = 5.0058 → 5
    assert float(sell.proceeds) == pytest.approx(4999.8)


def test_splitting_an_order_keeps_the_fee_at_one_per_lot():
    """兩份合併下單會多付手續費，拆單才符合「一份 = 1 元」的設計。"""
    from atrgrid.fees import split_buy_cost

    lot = max_shares_for_min_fee(16.10)  # 311 股
    merged = buy_cost(lot * 2, 16.10)
    split = split_buy_cost(2, lot, 16.10)

    assert merged.fee == 3      # 622 股一筆 → 3 元
    assert split.fee == 2       # 311 股兩筆 → 1 + 1 元
    assert split.gross == merged.gross


def test_split_sell_cost_charges_tax_per_order():
    from atrgrid.fees import split_sell_cost

    lot = max_shares_for_min_fee(61.80)
    split = split_sell_cost(3, lot, 61.80, "equity")
    single = sell_cost(lot, 61.80, "equity")

    assert split.fee == single.fee * 3
    assert split.tax == single.tax * 3
    assert split.shares == lot * 3


def test_split_sell_cost_is_tax_free_for_bonds():
    from atrgrid.fees import split_sell_cost

    lot = max_shares_for_min_fee(34.03)
    assert split_sell_cost(2, lot, 34.03, "bond").tax == 0
