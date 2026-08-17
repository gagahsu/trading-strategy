"""設定檔載入與驗證。

兩個檔案：

* ``config/settings.yaml``  ── 全域參數（手續費折數、各資產類別的網格參數、風控）
* ``config/portfolio.yaml`` ── 持股清單（代號、名稱、股數、平均成本、類別）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .fees import round_trip_cost_pct

VALID_CLASSES = {"equity", "bond", "leveraged"}
VALID_DRIFT_MODES = {"off", "up_only", "both"}


class ConfigError(Exception):
    """設定檔有問題時拋出。"""


@dataclass(frozen=True)
class GridParams:
    """單一資產類別（或單一標的覆寫）的網格參數。"""

    atr_period: int = 14
    #: 網格步長 = k * ATR
    atr_multiplier: float = 0.5
    #: 步長下限（占價格百分比），用來確保覆蓋來回交易成本
    min_step_pct: float = 0.8
    #: 步長上限（占價格百分比），避免極端波動時網格拉到永遠不觸發
    max_step_pct: float = 6.0
    #: 相對建檔股數，最多往下加碼幾份
    max_buy_rungs: int = 5
    #: 相對建檔股數，最多往上減碼幾份
    max_sell_rungs: int = 5
    #: 單日最多成交幾份（單邊）
    max_rungs_per_day: int = 2
    #: 當日相對前收的跳空超過 N 倍 ATR 就轉人工複核
    gap_atr_limit: float = 3.0
    #: 錨點漂移模式：off / up_only / both
    drift_mode: str = "up_only"
    #: 錨點每日往趨勢均線靠攏的比例
    drift_beta: float = 0.02
    #: 漂移參考的均線長度
    trend_ema_period: int = 60
    #: 是否允許虧損賣出（網格獲利了結預設不允許）
    allow_loss_sell: bool = False

    def validate(self, label: str) -> None:
        if self.atr_period < 2:
            raise ConfigError(f"{label}: atr_period 必須 >= 2")
        if self.atr_multiplier <= 0:
            raise ConfigError(f"{label}: atr_multiplier 必須 > 0")
        if self.min_step_pct <= 0:
            raise ConfigError(f"{label}: min_step_pct 必須 > 0")
        if self.max_step_pct <= self.min_step_pct:
            raise ConfigError(f"{label}: max_step_pct 必須 > min_step_pct")
        if self.max_buy_rungs < 0 or self.max_sell_rungs < 0:
            raise ConfigError(f"{label}: 階數上限不可為負")
        if self.max_rungs_per_day < 1:
            raise ConfigError(f"{label}: max_rungs_per_day 必須 >= 1")
        if self.drift_mode not in VALID_DRIFT_MODES:
            raise ConfigError(
                f"{label}: drift_mode 必須是 {sorted(VALID_DRIFT_MODES)} 之一"
            )
        if not 0.0 <= self.drift_beta <= 1.0:
            raise ConfigError(f"{label}: drift_beta 必須介於 0 與 1")
        if self.trend_ema_period < 2:
            raise ConfigError(f"{label}: trend_ema_period 必須 >= 2")

    def merged(self, overrides: dict[str, Any] | None) -> "GridParams":
        if not overrides:
            return self
        known = {f for f in self.__dataclass_fields__}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"未知的網格參數：{sorted(unknown)}")
        return replace(self, **overrides)


@dataclass(frozen=True)
class Settings:
    """全域設定。"""

    fee_discount: Decimal = Decimal("0.28")
    fee_minimum: int = 1
    #: 可動用現金（元）。買進會扣減，賣出會回補。
    cash: float = 0.0
    #: 現金水位低於此金額就停止買進
    cash_floor: float = 0.0
    #: 建議產生的時間（僅用於報表顯示）
    decision_time: str = "13:00"
    timezone: str = "Asia/Taipei"
    #: 日 K 資料超過幾天沒更新就視為過期
    max_data_staleness_days: int = 5
    #: 步長至少要是來回成本的幾倍，否則設定檢查會擋下
    min_step_cost_multiple: float = 3.0
    defaults: dict[str, GridParams] = field(default_factory=dict)

    def params_for(self, asset_class: str) -> GridParams:
        if asset_class not in self.defaults:
            raise ConfigError(f"settings.yaml 缺少資產類別 '{asset_class}' 的預設參數")
        return self.defaults[asset_class]


@dataclass(frozen=True)
class Holding:
    """一檔持股。"""

    ticker: str
    name: str
    asset_class: str
    shares: int
    avg_cost: float
    #: 代號是否已對照交易所資料驗證過。未驗證者不會產生下單建議。
    ticker_verified: bool = False
    enabled: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)
    #: 手動登記的除息，格式 [{"date": "2026-07-16", "amount": 0.35}]
    ex_dividends: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if self.asset_class not in VALID_CLASSES:
            raise ConfigError(
                f"{self.ticker}: asset_class '{self.asset_class}' 無效，"
                f"必須是 {sorted(VALID_CLASSES)} 之一"
            )
        if self.shares < 0:
            raise ConfigError(f"{self.ticker}: shares 不可為負")
        if self.avg_cost <= 0:
            raise ConfigError(f"{self.ticker}: avg_cost 必須 > 0")
        for entry in self.ex_dividends:
            if "date" not in entry or "amount" not in entry:
                raise ConfigError(
                    f"{self.ticker}: ex_dividends 每筆都需要 date 與 amount"
                )


@dataclass(frozen=True)
class Portfolio:
    holdings: list[Holding]

    def enabled(self) -> list[Holding]:
        return [h for h in self.holdings if h.enabled]

    def by_ticker(self, ticker: str) -> Holding:
        for holding in self.holdings:
            if holding.ticker == ticker:
                return holding
        raise KeyError(ticker)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"找不到設定檔：{path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 的最外層必須是對應（mapping）")
    return data


def load_settings(path: Path | str) -> Settings:
    raw = _read_yaml(Path(path))

    base = GridParams()
    defaults: dict[str, GridParams] = {}
    for asset_class, overrides in (raw.get("defaults") or {}).items():
        if asset_class not in VALID_CLASSES:
            raise ConfigError(f"settings.yaml: 未知的資產類別 '{asset_class}'")
        params = base.merged(overrides)
        params.validate(f"defaults.{asset_class}")
        defaults[asset_class] = params

    missing = VALID_CLASSES - set(defaults)
    if missing:
        raise ConfigError(f"settings.yaml 缺少資產類別預設值：{sorted(missing)}")

    fees = raw.get("fees") or {}
    risk = raw.get("risk") or {}

    settings = Settings(
        fee_discount=Decimal(str(fees.get("discount", "0.28"))),
        fee_minimum=int(fees.get("minimum", 1)),
        cash=float(risk.get("cash", 0.0)),
        cash_floor=float(risk.get("cash_floor", 0.0)),
        decision_time=str(raw.get("decision_time", "13:00")),
        timezone=str(raw.get("timezone", "Asia/Taipei")),
        max_data_staleness_days=int(risk.get("max_data_staleness_days", 5)),
        min_step_cost_multiple=float(risk.get("min_step_cost_multiple", 3.0)),
        defaults=defaults,
    )

    if not Decimal("0") < settings.fee_discount <= Decimal("1"):
        raise ConfigError("fees.discount 必須介於 0（不含）與 1 之間")
    if settings.fee_minimum < 0:
        raise ConfigError("fees.minimum 不可為負")
    if settings.cash < 0:
        raise ConfigError("risk.cash 不可為負")
    return settings


def load_portfolio(path: Path | str) -> Portfolio:
    raw = _read_yaml(Path(path))
    entries = raw.get("holdings")
    if not entries:
        raise ConfigError("portfolio.yaml 沒有任何 holdings")

    holdings: list[Holding] = []
    seen: set[str] = set()
    for entry in entries:
        ticker = str(entry["ticker"]).strip()
        if ticker in seen:
            raise ConfigError(f"重複的股票代號：{ticker}")
        seen.add(ticker)
        holding = Holding(
            ticker=ticker,
            name=str(entry.get("name", ticker)),
            asset_class=str(entry.get("class", "equity")),
            shares=int(entry.get("shares", 0)),
            avg_cost=float(entry.get("avg_cost", 0.0)),
            ticker_verified=bool(entry.get("ticker_verified", False)),
            enabled=bool(entry.get("enabled", True)),
            overrides=dict(entry.get("grid") or {}),
            ex_dividends=list(entry.get("ex_dividends") or []),
        )
        holding.validate()
        holdings.append(holding)
    return Portfolio(holdings)


def resolve_params(settings: Settings, holding: Holding) -> GridParams:
    """把類別預設值與個股覆寫合併成最終參數。"""
    params = settings.params_for(holding.asset_class).merged(holding.overrides)
    params.validate(holding.ticker)
    return params


def check_step_covers_costs(
    settings: Settings, holding: Holding, price: float
) -> str | None:
    """步長下限若沒有明顯高於來回交易成本就回傳警告字串。"""
    params = resolve_params(settings, holding)
    cost_pct = float(
        round_trip_cost_pct(
            price, holding.asset_class, settings.fee_discount, settings.fee_minimum
        )
    )
    required = cost_pct * settings.min_step_cost_multiple
    if params.min_step_pct < required:
        return (
            f"{holding.ticker} {holding.name}：min_step_pct {params.min_step_pct:.2f}% "
            f"低於來回成本 {cost_pct:.3f}% 的 {settings.min_step_cost_multiple:g} 倍"
            f"（需 >= {required:.2f}%）"
        )
    return None


def default_config_dir() -> Path:
    """允許用 ATRGRID_CONFIG_DIR 覆寫設定檔位置。"""
    env = os.environ.get("ATRGRID_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"
