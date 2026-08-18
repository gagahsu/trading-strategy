"""台股零股交易成本計算。

國泰證券手續費 2.8 折、最低 1 元、元以下無條件捨去。
本模組最重要的函式是 :func:`max_shares_for_min_fee` ── 找出「手續費仍只要 1 元」
的最大股數，也就是使用者操作的「一份」。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

#: 台股法定券商手續費率（未折扣）
BROKER_RATE = Decimal("0.001425")

#: 受益憑證（ETF）證券交易稅率
ETF_TAX_RATE = Decimal("0.001")

#: 債券 ETF 現行免徵證交稅
BOND_ETF_TAX_RATE = Decimal("0")

#: 個股（非 ETF）證券交易稅率，是 ETF 的 3 倍 —— 這是把個股跟 ETF 分開建類的
#: 主因，混在一起會少算稅。
STOCK_TAX_RATE = Decimal("0.003")


def _floor_int(value: Decimal) -> int:
    """元以下無條件捨去。"""
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def brokerage_fee(
    amount: Decimal | float | int,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> int:
    """券商手續費。

    ``amount`` 為成交金額。折扣後小數點無條件捨去，再套用最低收費。
    金額為 0 時不收費（沒有成交就沒有手續費）。
    """
    amount = Decimal(str(amount))
    if amount <= 0:
        return 0
    raw = amount * BROKER_RATE * Decimal(str(discount))
    return max(_floor_int(raw), minimum)


def transaction_tax(amount: Decimal | float | int, asset_class: str) -> int:
    """賣出證交稅。債券 ETF 免徵、個股 0.3%、其餘（股票型/槓桿型 ETF）0.1%，元以下捨去。"""
    amount = Decimal(str(amount))
    if amount <= 0:
        return 0
    if asset_class == "bond":
        rate = BOND_ETF_TAX_RATE
    elif asset_class == "stock":
        rate = STOCK_TAX_RATE
    else:
        rate = ETF_TAX_RATE
    return _floor_int(amount * rate)


def max_shares_for_min_fee(
    price: Decimal | float | int,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> int:
    """手續費仍等於最低收費（預設 1 元）時可買進的最大股數 ── 使用者的「一份」。

    以 2.8 折為例，實際費率 0.001425 * 0.28 = 0.000399。手續費要維持 1 元，
    成交金額必須滿足 ``floor(金額 * 0.000399) <= 1``，也就是金額 < 5012.53 元。

    手續費隨股數單調遞增，所以用二分搜尋找臨界點。若連 1 股都超過門檻
    （例如超高價標的），回傳 1 ── 一份至少是 1 股。
    """
    price = Decimal(str(price))
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")

    def fee_of(shares: int) -> int:
        return brokerage_fee(price * shares, discount, minimum)

    if fee_of(1) > minimum:
        return 1

    # 先指數放大找到一個「已超過」的上界，再二分收斂。
    hi = 1
    while fee_of(hi) <= minimum:
        hi *= 2
        if hi > 10_000_000:  # 現實中不可能，純粹防呆
            break

    lo = hi // 2  # fee_of(lo) <= minimum
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fee_of(mid) <= minimum:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class TradeCost:
    """一筆零股買賣的完整成本試算。"""

    shares: int
    price: Decimal
    gross: Decimal
    fee: int
    tax: int

    @property
    def net(self) -> Decimal:
        """買進為實付總額；賣出請用 :attr:`proceeds`。"""
        return self.gross + self.fee + self.tax

    @property
    def proceeds(self) -> Decimal:
        """賣出實收金額。"""
        return self.gross - self.fee - self.tax

    @property
    def cost_pct(self) -> Decimal:
        if self.gross == 0:
            return Decimal(0)
        return (Decimal(self.fee + self.tax) / self.gross) * 100


def buy_cost(
    shares: int,
    price: Decimal | float | int,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> TradeCost:
    price = Decimal(str(price))
    gross = price * shares
    return TradeCost(shares, price, gross, brokerage_fee(gross, discount, minimum), 0)


def sell_cost(
    shares: int,
    price: Decimal | float | int,
    asset_class: str,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> TradeCost:
    price = Decimal(str(price))
    gross = price * shares
    return TradeCost(
        shares,
        price,
        gross,
        brokerage_fee(gross, discount, minimum),
        transaction_tax(gross, asset_class),
    )


def split_buy_cost(
    rungs: int,
    lot: int,
    price: Decimal | float | int,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> TradeCost:
    """買進 ``rungs`` 份時，**拆成 rungs 筆各 lot 股**下單的合計成本。

    這件事很重要：一份的定義就是「手續費剛好 1 元的最大股數」，所以兩份
    合併成一筆下單會踩過門檻（例如 622 股要 3 元），拆成兩筆各 311 股則
    只要 1+1 = 2 元。系統一律以拆單為前提試算。
    """
    price = Decimal(str(price))
    per_order = brokerage_fee(price * lot, discount, minimum)
    return TradeCost(
        shares=lot * rungs,
        price=price,
        gross=price * lot * rungs,
        fee=per_order * rungs,
        tax=0,
    )


def split_sell_cost(
    rungs: int,
    lot: int,
    price: Decimal | float | int,
    asset_class: str,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> TradeCost:
    """賣出 ``rungs`` 份、拆成 rungs 筆各 lot 股的合計成本。

    證交稅同樣逐筆計算後捨去，所以拆單的稅額可能比合併下單少一點。
    """
    price = Decimal(str(price))
    per_fee = brokerage_fee(price * lot, discount, minimum)
    per_tax = transaction_tax(price * lot, asset_class)
    return TradeCost(
        shares=lot * rungs,
        price=price,
        gross=price * lot * rungs,
        fee=per_fee * rungs,
        tax=per_tax * rungs,
    )


def round_trip_cost_pct(
    price: Decimal | float | int,
    asset_class: str,
    discount: Decimal | float | str = "0.28",
    minimum: int = 1,
) -> Decimal:
    """一買一賣「一份」的來回成本，以百分比表示。

    這是網格步長的下限依據：步長若不明顯大於來回成本，網格只是在幫券商和
    國庫打工。
    """
    shares = max_shares_for_min_fee(price, discount, minimum)
    buy = buy_cost(shares, price, discount, minimum)
    sell = sell_cost(shares, price, asset_class, discount, minimum)
    if buy.gross == 0:
        return Decimal(0)
    total = Decimal(buy.fee + sell.fee + sell.tax)
    return (total / buy.gross) * 100
