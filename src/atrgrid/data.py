"""行情資料來源。

兩個 provider：

* :class:`TwseProvider` ── 證交所公開 API（日 K + 盤中即時價 + 名稱驗證）
* :class:`CsvProvider`  ── 讀本地 CSV，離線測試與回測用

證交所 API 沒有正式的服務水準保證，也有流量限制，所以每次呼叫之間會停頓，
且日 K 會快取到 ``data/cache/``。
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path

from .indicators import Bar

USER_AGENT = "Mozilla/5.0 (compatible; atrgrid/1.0; +https://github.com/)"
TWSE_DAILY = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TWSE_QUOTE = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
TPEX_DAILY = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"


class DataError(Exception):
    """抓不到資料或資料格式不符時拋出。"""


class PriceProvider(ABC):
    """行情來源介面。"""

    @abstractmethod
    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        """回傳依日期遞增排序的已收盤日 K。"""

    @abstractmethod
    def live_price(self, ticker: str) -> float:
        """回傳盤中即時價（或最近成交價）。"""

    def security_name(self, ticker: str) -> str | None:
        """回傳交易所登記的證券名稱，用於驗證代號。取不到時回傳 None。"""
        return None


# --------------------------------------------------------------------- CSV


class CsvProvider(PriceProvider):
    """從 ``<root>/<ticker>.csv`` 讀取日 K。

    欄位：``date,open,high,low,close,volume``（有表頭）。
    即時價預設取最後一根的收盤，可用 ``live_overrides`` 覆寫。
    """

    def __init__(
        self, root: Path | str, live_overrides: dict[str, float] | None = None
    ) -> None:
        self.root = Path(root)
        self.live_overrides = live_overrides or {}

    def _path(self, ticker: str) -> Path:
        return self.root / f"{ticker}.csv"

    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        path = self._path(ticker)
        if not path.exists():
            raise DataError(f"找不到 CSV：{path}")
        bars: list[Bar] = []
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                bars.append(
                    Bar(
                        date=row["date"],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0),
                    )
                )
        bars.sort(key=lambda b: b.date)
        return bars

    def live_price(self, ticker: str) -> float:
        if ticker in self.live_overrides:
            return self.live_overrides[ticker]
        bars = self.daily_bars(ticker)
        if not bars:
            raise DataError(f"{ticker} 沒有任何 K 棒")
        return bars[-1].close

    def security_name(self, ticker: str) -> str | None:
        return None


# -------------------------------------------------------------------- TWSE


def _http_get_json(url: str, timeout: int = 20, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise DataError(f"讀取 {url} 失敗：{last}")


class TwseProvider(PriceProvider):
    """證交所（上市）行情。"""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        throttle: float = 3.0,
        cache_ttl_hours: float = 12.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.throttle = throttle
        self.cache_ttl_hours = cache_ttl_hours
        self._last_call = 0.0

    def _wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self._last_call = time.time()

    # ------------------------------------------------------------ 日 K
    def _fetch_month(self, ticker: str, month_start: date) -> list[Bar]:
        url = (
            f"{TWSE_DAILY}?date={month_start.strftime('%Y%m%d')}"
            f"&stockNo={ticker}&response=json"
        )
        self._wait()
        payload = _http_get_json(url)
        if payload.get("stat") != "OK":
            raise DataError(
                f"{ticker} {month_start:%Y-%m}：證交所回應 {payload.get('stat')}"
            )
        bars: list[Bar] = []
        for row in payload.get("data") or []:
            try:
                bars.append(_parse_twse_row(row))
            except (ValueError, IndexError):
                continue  # 無成交日的資料列會有 "--"，跳過
        return bars

    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        cached = self._read_cache(ticker)
        if cached is not None:
            return cached

        today = date.today()
        bars: dict[str, Bar] = {}
        cursor = today.replace(day=1)
        for _ in range(months):
            try:
                for bar in self._fetch_month(ticker, cursor):
                    bars[bar.date] = bar
            except DataError:
                pass  # 個別月份失敗不致命，繼續往前抓
            cursor = (cursor - timedelta(days=1)).replace(day=1)

        if not bars:
            raise DataError(f"{ticker}：抓不到任何日 K，請確認代號是否正確")

        # 今天的 K 棒尚未收盤，排除以免污染 ATR。
        result = sorted(
            (bar for bar in bars.values() if bar.date < today.isoformat()),
            key=lambda b: b.date,
        )
        self._write_cache(ticker, result)
        return result

    # ------------------------------------------------------------ 即時價
    def live_price(self, ticker: str) -> float:
        payload = self._quote(ticker)
        for key in ("z", "o", "y"):  # 成交價 → 開盤價 → 昨收
            raw = payload.get(key)
            if raw and raw not in ("-", "--"):
                return float(raw)
        # 盤中無成交時，退而取最佳五檔買賣價的中價。
        bid = (payload.get("b") or "").split("_")[0]
        ask = (payload.get("a") or "").split("_")[0]
        if bid and ask and bid != "-" and ask != "-":
            return (float(bid) + float(ask)) / 2
        raise DataError(f"{ticker}：即時報價沒有可用價格欄位")

    def security_name(self, ticker: str) -> str | None:
        try:
            return (self._quote(ticker).get("n") or "").strip() or None
        except DataError:
            return None

    def _quote(self, ticker: str) -> dict:
        self._wait()
        url = f"{TWSE_QUOTE}?ex_ch=tse_{ticker}.tw&json=1&delay=0&_={int(time.time() * 1000)}"
        payload = _http_get_json(url)
        entries = payload.get("msgArray") or []
        if not entries:
            raise DataError(f"{ticker}：證交所即時報價無資料（代號可能有誤或非上市）")
        return entries[0]

    # -------------------------------------------------------------- 快取
    def _cache_path(self, ticker: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{ticker}.json"

    def _read_cache(self, ticker: str) -> list[Bar] | None:
        path = self._cache_path(ticker)
        if path is None or not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cache_ttl_hours:
            return None
        with path.open(encoding="utf-8") as handle:
            return [Bar(**row) for row in json.load(handle)]

    def _write_cache(self, ticker: str, bars: list[Bar]) -> None:
        path = self._cache_path(ticker)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([bar.__dict__ for bar in bars], handle, ensure_ascii=False)


def _parse_twse_row(row: list[str]) -> Bar:
    """把證交所的一列日 K 轉成 :class:`Bar`。

    日期是民國年（``115/08/15``），數字帶千分位逗號。
    """
    roc_year, month, day = row[0].split("/")
    iso = f"{int(roc_year) + 1911:04d}-{int(month):02d}-{int(day):02d}"

    def num(value: str) -> float:
        return float(value.replace(",", "").strip())

    return Bar(
        date=iso,
        volume=num(row[1]),
        open=num(row[3]),
        high=num(row[4]),
        low=num(row[5]),
        close=num(row[6]),
    )


def make_provider(
    kind: str,
    csv_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    live_overrides: dict[str, float] | None = None,
) -> PriceProvider:
    if kind == "twse":
        return TwseProvider(cache_dir=cache_dir)
    if kind == "csv":
        if csv_dir is None:
            raise DataError("csv provider 需要指定 --csv-dir")
        return CsvProvider(csv_dir, live_overrides=live_overrides)
    raise DataError(f"未知的 provider：{kind}")


def export_csv(bars: list[Bar], path: Path | str) -> None:
    """把日 K 存成 CSV，方便離線回測。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow(
                [bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume]
            )


def parse_date(value: str) -> str:
    """接受 YYYY-MM-DD 或 YYYYMMDD，統一輸出 ISO 格式。"""
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"無法解析日期：{value}")
