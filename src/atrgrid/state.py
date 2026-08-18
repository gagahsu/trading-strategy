"""網格狀態的持久化。

狀態是這套系統的記憶：錨點在哪、加減碼到第幾階、每一份的買進成本是多少。
沒有狀態，網格每天都會從頭開始，也就不是網格了。

存成 JSON（``state/state.json``），可讀、可手改、可進版控。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

STATE_VERSION = 2


@dataclass
class Lot:
    """一筆買進紀錄，賣出時用來配對計算實現損益。"""

    date: str
    price: float
    shares: int
    #: initial = 建檔時的既有持股；grid = 網格買進
    source: str = "grid"

    def cost(self) -> float:
        return self.price * self.shares


@dataclass
class Trade:
    """已記錄的成交。"""

    date: str
    ticker: str
    action: str  # BUY / SELL
    shares: int
    price: float
    fee: int
    tax: int
    rungs: int
    realized_pnl: float = 0.0
    note: str = ""


@dataclass
class Position:
    """單一標的的網格狀態。"""

    ticker: str
    shares: int
    #: 網格基準價
    anchor: float
    #: 相對建檔股數的階數，正數代表已往下加碼
    rung: int = 0
    #: 建檔時的股數，作為階數的原點
    baseline_shares: int = 0
    lots: list[Lot] = field(default_factory=list)
    realized_pnl: float = 0.0
    last_trade_date: str | None = None
    #: 已套用過的除息日期，避免重複調整錨點
    applied_ex_dividends: list[str] = field(default_factory=list)
    #: 最後一次套用錨點漂移的日期，確保一天只漂移一次
    last_drift_date: str | None = None

    def total_lot_shares(self) -> int:
        return sum(lot.shares for lot in self.lots)

    def average_cost(self) -> float:
        total = self.total_lot_shares()
        if total == 0:
            return 0.0
        return sum(lot.cost() for lot in self.lots) / total

    def apply_buy(self, trade_date: str, price: float, shares: int, rungs: int) -> None:
        self.shares += shares
        self.lots.append(Lot(date=trade_date, price=price, shares=shares))
        self.rung += rungs
        self.last_trade_date = trade_date

    def apply_sell(
        self, trade_date: str, price: float, shares: int, rungs: int
    ) -> float:
        """後進先出配對賣出，回傳毛實現損益（未扣手續費與稅）。"""
        remaining = shares
        proceeds_basis = 0.0
        while remaining > 0 and self.lots:
            lot = self.lots[-1]
            take = min(lot.shares, remaining)
            proceeds_basis += lot.price * take
            lot.shares -= take
            remaining -= take
            if lot.shares == 0:
                self.lots.pop()
        if remaining > 0:
            # 狀態與實際持股不一致時的保險絲：用剩餘部位的均價補齊。
            fallback = self.average_cost() or price
            proceeds_basis += fallback * remaining
        gross_pnl = price * shares - proceeds_basis
        self.shares -= shares
        self.rung -= rungs
        self.realized_pnl += gross_pnl
        self.last_trade_date = trade_date
        return gross_pnl

    def peek_sell_basis(self, shares: int) -> float:
        """試算賣出 ``shares`` 股的成本基礎，不改動狀態。"""
        remaining = shares
        basis = 0.0
        for lot in reversed(self.lots):
            if remaining <= 0:
                break
            take = min(lot.shares, remaining)
            basis += lot.price * take
            remaining -= take
        if remaining > 0:
            basis += (self.average_cost() or 0.0) * remaining
        return basis


@dataclass
class State:
    version: int = STATE_VERSION
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    last_run_date: str | None = None

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cash": round(self.cash, 2),
            "last_run_date": self.last_run_date,
            "positions": {
                ticker: {
                    **asdict(position),
                    "anchor": round(position.anchor, 4),
                    "realized_pnl": round(position.realized_pnl, 2),
                }
                for ticker, position in sorted(self.positions.items())
            },
            "trades": [asdict(trade) for trade in self.trades],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        version = int(data.get("version", 1))
        if version > STATE_VERSION:
            raise ValueError(
                f"狀態檔版本 {version} 比程式支援的 {STATE_VERSION} 還新，請先更新程式"
            )
        positions: dict[str, Position] = {}
        for ticker, raw in (data.get("positions") or {}).items():
            lots = [Lot(**lot) for lot in raw.get("lots", [])]
            positions[ticker] = Position(
                ticker=ticker,
                shares=int(raw["shares"]),
                anchor=float(raw["anchor"]),
                rung=int(raw.get("rung", 0)),
                baseline_shares=int(raw.get("baseline_shares", raw["shares"])),
                lots=lots,
                realized_pnl=float(raw.get("realized_pnl", 0.0)),
                last_trade_date=raw.get("last_trade_date"),
                applied_ex_dividends=list(raw.get("applied_ex_dividends") or []),
                last_drift_date=raw.get("last_drift_date"),
            )
        return cls(
            version=STATE_VERSION,
            cash=float(data.get("cash", 0.0)),
            positions=positions,
            trades=[Trade(**t) for t in (data.get("trades") or [])],
            last_run_date=data.get("last_run_date"),
        )


def load_state(path: Path | str) -> State:
    path = Path(path)
    if not path.exists():
        return State()
    with path.open(encoding="utf-8") as handle:
        return State.from_dict(json.load(handle))


def save_state(state: State, path: Path | str) -> None:
    """原子寫入，並保留一份 ``.bak``。狀態檔壞掉等於整套網格失憶。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def init_state(
    holdings: Iterable[Any],
    prices: dict[str, float],
    cash: float,
    as_of: str | None = None,
) -> State:
    """從 portfolio.yaml 與一組起始價格建立初始狀態。

    既有持股記成一筆 ``initial`` 成本批次；錨點就設在建檔當下的價格 ── 網格從
    「此時此刻」開始運作，不去追溯過去的進出。
    """
    as_of = as_of or date.today().isoformat()
    positions: dict[str, Position] = {}
    for holding in holdings:
        price = prices.get(holding.ticker)
        if price is None:
            raise KeyError(f"缺少 {holding.ticker} 的建檔價格")
        lots = []
        if holding.shares > 0:
            lots.append(
                Lot(
                    date=as_of,
                    price=holding.avg_cost,
                    shares=holding.shares,
                    source="initial",
                )
            )
        positions[holding.ticker] = Position(
            ticker=holding.ticker,
            shares=holding.shares,
            anchor=price,
            rung=0,
            baseline_shares=holding.shares,
            lots=lots,
        )
    return State(cash=cash, positions=positions, last_run_date=None)


def add_position(state: State, holding: Any, price: float, as_of: str | None = None) -> Position:
    """幫既有狀態加一檔新持股的部位，不動其他標的與現金。

    跟 :func:`init_state` 同一套錨點邏輯：錨點設在此刻的市價，既有股數（如果
    有）記成一筆 ``initial`` 成本批次。
    """
    if holding.ticker in state.positions:
        raise ValueError(f"{holding.ticker} 已經有部位了")
    as_of = as_of or date.today().isoformat()
    lots = []
    if holding.shares > 0:
        lots.append(
            Lot(date=as_of, price=holding.avg_cost, shares=holding.shares, source="initial")
        )
    position = Position(
        ticker=holding.ticker,
        shares=holding.shares,
        anchor=price,
        rung=0,
        baseline_shares=holding.shares,
        lots=lots,
    )
    state.positions[holding.ticker] = position
    return position
