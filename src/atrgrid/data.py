"""行情資料來源。

* :class:`YahooProvider`   ── Yahoo Finance，一次呼叫同時給日 K 與盤中價
* :class:`FinMindProvider` ── FinMind 開放資料（日 K；免費方案無盤中報價）
* :class:`TwseProvider`    ── 證交所公開 API（日 K、盤中價、名稱驗證）
* :class:`CsvProvider`     ── 讀本地 CSV，離線測試與回測用
* :class:`ChainProvider`   ── 依序嘗試多個來源，第一個成功的採用

這些都不是有服務水準保證的 API，也都有流量限制，所以每次呼叫之間會停頓，
且日 K 會快取到 ``data/cache/``。
"""

from __future__ import annotations

import csv
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .indicators import Bar

try:  # 用 certifi 的憑證庫，繞開部分機器上系統憑證鏈缺 SKI 欄位導致的驗證失敗
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - certifi 未安裝時退回系統預設驗證
    _SSL_CONTEXT = ssl.create_default_context()

try:  # Python 3.9+ 標準庫；缺 tzdata 的精簡環境則退回固定偏移
    from zoneinfo import ZoneInfo

    TAIPEI = ZoneInfo("Asia/Taipei")
except Exception:  # pragma: no cover - 視執行環境而定
    TAIPEI = timezone(timedelta(hours=8))

USER_AGENT = "Mozilla/5.0 (compatible; atrgrid/1.0; +https://github.com/)"
TWSE_DAILY = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TWSE_QUOTE = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
TWSE_EXRIGHT = "https://www.twse.com.tw/exchangeReport/TWT48U"
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

    def dividends(self, ticker: str) -> list[dict]:
        """回傳來源已知的除息事件 ``[{"date": "YYYY-MM-DD", "amount": float}]``。

        不是每個來源都有這項資料，取不到就回傳空列表 —— 呼叫端仍要把它當
        「參考」而非權威資料：抓到的股利事件要人工核對金額與日期後才登記
        進 ``portfolio.yaml``（見 :func:`atrgrid.config.add_ex_dividend`）。
        """
        return []


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
            with urllib.request.urlopen(
                request, timeout=timeout, context=_SSL_CONTEXT
            ) as response:
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
        self._exright_cache: dict[str, list[list[str]]] = {}

    def _wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self._last_call = time.time()

    # ------------------------------------------------------------ 除權息
    def dividends(self, ticker: str) -> list[dict]:
        """查「除權除息預告表」（TWT48U），抓現金股利 > 0 的列。

        這張表是證交所當天就會登的官方資料，比 Yahoo 的 dividend event
        （常常延遲一兩天才出現）即時，適合當天 13:00 跑 advise 時抓當天
        剛發生的除息。這張表本身就是近期快照，``date=`` 查詢參數證交所端
        其實沒在用（不管填哪天都回同一份），所以只查一次，不要逐日回溯，
        免得同一筆事件被重複計入。
        """
        out: list[dict] = []
        seen: set[tuple[str, float]] = set()
        for row in self._exright_day(date.today()):
            if len(row) < 8 or row[1] != ticker:
                continue
            if row[3] not in ("息", "權息"):
                continue
            try:
                amount = float(row[7])
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            try:
                ex_date = _parse_roc_date(row[0])
            except ValueError:
                continue
            amount = round(amount, 4)
            if (ex_date, amount) in seen:
                continue
            seen.add((ex_date, amount))
            out.append({"date": ex_date, "amount": amount})
        out.sort(key=lambda d: d["date"])
        return out

    def _exright_day(self, day: date) -> list[list[str]]:
        key = day.strftime("%Y%m%d")
        if key not in self._exright_cache:
            self._wait()
            url = f"{TWSE_EXRIGHT}?response=json&date={key}"
            try:
                payload = _http_get_json(url)
            except DataError:
                payload = {}
            rows = payload.get("data") or [] if payload.get("stat") == "OK" else []
            self._exright_cache[key] = rows
        return self._exright_cache[key]

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


def _parse_roc_date(value: str) -> str:
    """把民國日期（``115年08月18日``）轉成 ISO（``2026-08-18``）。"""
    match = re.match(r"(\d+)年(\d+)月(\d+)日", value.strip())
    if not match:
        raise ValueError(f"無法解析民國日期：{value}")
    roc_year, month, day = match.groups()
    return f"{int(roc_year) + 1911:04d}-{int(month):02d}-{int(day):02d}"


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


#: 可用的行情來源名稱
PROVIDER_KINDS = ("yahoo", "finmind", "twse", "auto", "csv")


def make_provider(
    kind: str,
    csv_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    live_overrides: dict[str, float] | None = None,
    market: dict[str, str] | None = None,
    finmind_token: str | None = None,
) -> PriceProvider:
    """依名稱建立行情來源。

    ``auto`` 會串起 Yahoo → 證交所 → FinMind：Yahoo 一次給日 K 與盤中價，
    證交所補上 Yahoo 查不到的台股冷門標的，FinMind 作為最後備援。
    """
    if kind == "yahoo":
        return YahooProvider(cache_dir=cache_dir, market=market)
    if kind == "finmind":
        return FinMindProvider(token=finmind_token, cache_dir=cache_dir)
    if kind == "twse":
        return TwseProvider(cache_dir=cache_dir)
    if kind == "auto":
        return ChainProvider([
            YahooProvider(cache_dir=cache_dir, market=market),
            TwseProvider(cache_dir=cache_dir),
            FinMindProvider(token=finmind_token, cache_dir=cache_dir),
        ])
    if kind == "csv":
        if csv_dir is None:
            raise DataError("csv provider 需要指定 --csv-dir")
        return CsvProvider(csv_dir, live_overrides=live_overrides)
    raise DataError(f"未知的 provider：{kind}（可用：{', '.join(PROVIDER_KINDS)}）")


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


# ------------------------------------------------------------------- Yahoo


class YahooProvider(PriceProvider):
    """Yahoo Finance 行情。

    ``/v8/finance/chart`` 一次呼叫就同時給出日 K 與盤中最新價，是這套系統
    在 13:00 取價最省事的來源。

    台股代號要加後綴：上市 ``.TW``、上櫃 ``.TWO``。若沒指定市場別，兩個都試。
    """

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        throttle: float = 1.0,
        cache_ttl_hours: float = 12.0,
        market: dict[str, str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.throttle = throttle
        self.cache_ttl_hours = cache_ttl_hours
        self.market = market or {}
        self._last_call = 0.0
        self._chart_cache: dict[str, dict] = {}

    # ------------------------------------------------------------ 內部
    def _suffixes(self, ticker: str) -> list[str]:
        market = self.market.get(ticker)
        if market == "otc":
            return [".TWO"]
        if market == "listed":
            return [".TW"]
        return [".TW", ".TWO"]

    def _wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self._last_call = time.time()

    def _chart(self, ticker: str, rng: str = "2y", events: str | None = None) -> dict:
        key = f"{ticker}:{rng}:{events or ''}"
        if key in self._chart_cache:
            return self._chart_cache[key]

        errors: list[str] = []
        for suffix in self._suffixes(ticker):
            url = f"{self.BASE}/{ticker}{suffix}?range={rng}&interval=1d"
            if events:
                url += f"&events={events}"
            self._wait()
            try:
                payload = _http_get_json(url)
            except DataError as exc:
                errors.append(f"{suffix}: {exc}")
                continue
            chart = payload.get("chart") or {}
            if chart.get("error"):
                errors.append(f"{suffix}: {chart['error']}")
                continue
            results = chart.get("result") or []
            if not results:
                errors.append(f"{suffix}: 無資料")
                continue
            self._chart_cache[key] = results[0]
            return results[0]
        raise DataError(f"{ticker}：Yahoo 查無資料（{'; '.join(errors)}）")

    # ------------------------------------------------------------ 介面
    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        cached = _read_bar_cache(self.cache_dir, ticker, self.cache_ttl_hours)
        if cached is not None:
            return cached

        rng = "2y" if months > 12 else "1y"
        result = self._chart(ticker, rng)
        stamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]

        today = date.today().isoformat()
        bars: list[Bar] = []
        for i, stamp in enumerate(stamps):
            row = [quote.get(k, [])[i] if i < len(quote.get(k, [])) else None
                   for k in ("open", "high", "low", "close", "volume")]
            if any(v is None for v in row[:4]):
                continue  # 停牌或無成交
            open_, high, low, close, volume = row
            iso = _epoch_to_taipei_date(stamp)
            if iso >= today:
                continue  # 今天的 K 棒還沒收，不能拿來算 ATR
            bars.append(
                Bar(
                    date=iso,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume or 0),
                )
            )
        if not bars:
            raise DataError(f"{ticker}：Yahoo 回傳的日 K 是空的")
        bars.sort(key=lambda b: b.date)
        _write_bar_cache(self.cache_dir, ticker, bars)
        return bars

    def live_price(self, ticker: str) -> float:
        meta = self._chart(ticker).get("meta") or {}
        for key in ("regularMarketPrice", "previousClose", "chartPreviousClose"):
            value = meta.get(key)
            if value:
                return float(value)
        raise DataError(f"{ticker}：Yahoo 沒有回傳可用的價格欄位")

    def security_name(self, ticker: str) -> str | None:
        meta = self._chart(ticker).get("meta") or {}
        name = meta.get("longName") or meta.get("shortName")
        return str(name).strip() if name else None

    def dividends(self, ticker: str) -> list[dict]:
        result = self._chart(ticker, rng="2y", events="div")
        events = ((result.get("events") or {}).get("dividends") or {}).values()
        out = [
            {"date": _epoch_to_taipei_date(e["date"]), "amount": round(float(e["amount"]), 4)}
            for e in events
            if e.get("date") is not None and e.get("amount") is not None
        ]
        out.sort(key=lambda d: d["date"])
        return out


# ----------------------------------------------------------------- FinMind


class FinMindProvider(PriceProvider):
    """FinMind 開放資料 API。

    免費方案有每小時請求上限，且日線資料為收盤後更新 —— 盤中取價請搭配
    Yahoo 或證交所。註冊後可在 https://finmindtrade.com 取得 token，
    以環境變數 ``FINMIND_TOKEN`` 提供。
    """

    BASE = "https://api.finmindtrade.com/api/v4/data"

    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path | str | None = None,
        throttle: float = 1.0,
        cache_ttl_hours: float = 12.0,
    ) -> None:
        self.token = token or os.environ.get("FINMIND_TOKEN", "")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.throttle = throttle
        self.cache_ttl_hours = cache_ttl_hours
        self._last_call = 0.0
        self._names: dict[str, str] | None = None

    def _wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self._last_call = time.time()

    def _query(self, dataset: str, **params: str) -> list[dict]:
        query = {"dataset": dataset, **params}
        if self.token:
            query["token"] = self.token
        url = f"{self.BASE}?{urllib.parse.urlencode(query)}"
        self._wait()
        payload = _http_get_json(url)
        if payload.get("status") not in (200, "200", None):
            raise DataError(
                f"FinMind {dataset} 回應 {payload.get('status')}：{payload.get('msg')}"
            )
        return payload.get("data") or []

    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        cached = _read_bar_cache(self.cache_dir, ticker, self.cache_ttl_hours)
        if cached is not None:
            return cached

        start = (date.today() - timedelta(days=int(months * 31))).isoformat()
        rows = self._query(
            "TaiwanStockPrice", data_id=ticker, start_date=start
        )
        today = date.today().isoformat()
        bars: list[Bar] = []
        for row in rows:
            iso = str(row["date"])
            if iso >= today:
                continue
            try:
                # FinMind 用 max / min 而非 high / low
                bars.append(
                    Bar(
                        date=iso,
                        open=float(row["open"]),
                        high=float(row["max"]),
                        low=float(row["min"]),
                        close=float(row["close"]),
                        volume=float(row.get("Trading_Volume") or 0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not bars:
            raise DataError(f"{ticker}：FinMind 沒有回傳日 K（確認代號與 token）")
        bars.sort(key=lambda b: b.date)
        _write_bar_cache(self.cache_dir, ticker, bars)
        return bars

    def live_price(self, ticker: str) -> float:
        """FinMind 免費方案沒有盤中報價，回傳最近一個收盤價。"""
        bars = self.daily_bars(ticker, months=2)
        return bars[-1].close

    def security_name(self, ticker: str) -> str | None:
        if self._names is None:
            try:
                rows = self._query("TaiwanStockInfo")
            except DataError:
                return None
            self._names = {
                str(r.get("stock_id")): str(r.get("stock_name", "")).strip()
                for r in rows
            }
        return self._names.get(ticker) or None


# ------------------------------------------------------------------- 串接


class ChainProvider(PriceProvider):
    """依序嘗試多個來源，第一個成功的就採用。

    實務上很有用：日 K 用 FinMind 或 Yahoo，盤中價用證交所，任何一家掛掉
    還有備援。所有來源都失敗時，錯誤訊息會列出每一家的原因。
    """

    def __init__(self, providers: list[PriceProvider]) -> None:
        if not providers:
            raise DataError("ChainProvider 至少需要一個來源")
        self.providers = providers

    def _try(self, method: str, ticker: str, *args):
        errors: list[str] = []
        for provider in self.providers:
            try:
                return getattr(provider, method)(ticker, *args)
            except (DataError, KeyError, ValueError, TypeError) as exc:
                errors.append(f"{type(provider).__name__}: {exc}")
        raise DataError(f"{ticker}：所有來源皆失敗（{'; '.join(errors)}）")

    def daily_bars(self, ticker: str, months: int = 12) -> list[Bar]:
        return self._try("daily_bars", ticker, months)

    def live_price(self, ticker: str) -> float:
        return self._try("live_price", ticker)

    def security_name(self, ticker: str) -> str | None:
        for provider in self.providers:
            try:
                name = provider.security_name(ticker)
            except DataError:
                continue
            if name:
                return name
        return None

    def dividends(self, ticker: str) -> list[dict]:
        """合併所有來源的結果（依日期去重），不是第一個有結果就採用。

        來源之間對「除息事件」的涵蓋範圍不互補：Yahoo 涵蓋歷史久但當天
        常延遲登錄，TWSE 只有近期快照但當天就有。用「第一個成功就採用」
        的規則（daily_bars/live_price 用的那種）會讓 Yahoo 一有舊資料就
        整個短路，永遠查不到 TWSE 才有的當日事件。
        """
        merged: dict[str, dict] = {}
        for provider in self.providers:
            try:
                events = provider.dividends(ticker)
            except DataError:
                continue
            for event in events:
                merged.setdefault(str(event["date"]), event)
        return sorted(merged.values(), key=lambda d: d["date"])


# --------------------------------------------------------------- 共用快取


def _read_bar_cache(
    cache_dir: Path | None, ticker: str, ttl_hours: float
) -> list[Bar] | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{ticker}.json"
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) / 3600 > ttl_hours:
        return None
    with path.open(encoding="utf-8") as handle:
        return [Bar(**row) for row in json.load(handle)]


def _write_bar_cache(cache_dir: Path | None, ticker: str, bars: list[Bar]) -> None:
    if cache_dir is None:
        return
    path = Path(cache_dir) / f"{ticker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([bar.__dict__ for bar in bars], handle, ensure_ascii=False)


def _epoch_to_taipei_date(stamp: int | float) -> str:
    """Yahoo 的時間戳是 UTC epoch，要換算成台北當地日期。"""
    return datetime.fromtimestamp(int(stamp), tz=TAIPEI).date().isoformat()
