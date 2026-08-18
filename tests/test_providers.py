"""行情來源的解析邏輯測試。

沙箱連不到 Yahoo / FinMind，所以這裡用**錄製的回應格式**驗證解析與過濾，
不驗證線上服務本身。欄位名稱若被上游改掉，這些測試不會發現 ——
那要靠 `atrgrid verify-tickers` 實際連線才會暴露。
"""

from datetime import date, timedelta

import pytest

from atrgrid import data as D
from atrgrid.data import (
    ChainProvider,
    DataError,
    FinMindProvider,
    PriceProvider,
    YahooProvider,
    make_provider,
)
from atrgrid.indicators import Bar


def _epoch(iso: str) -> int:
    """台北當地日期 13:30 收盤的 epoch 秒。"""
    from datetime import datetime

    d = date.fromisoformat(iso)
    return int(
        datetime(d.year, d.month, d.day, 13, 30, tzinfo=D.TAIPEI).timestamp()
    )


def yahoo_payload(dates, closes, live=None, name="元大台灣50"):
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "0050.TW",
                        "regularMarketPrice": live,
                        "previousClose": closes[-1],
                        "longName": name,
                    },
                    "timestamp": [_epoch(d) for d in dates],
                    "indicators": {
                        "quote": [
                            {
                                "open": [c - 0.5 for c in closes],
                                "high": [c + 1.0 for c in closes],
                                "low": [c - 1.0 for c in closes],
                                "close": list(closes),
                                "volume": [1_000_000] * len(closes),
                            }
                        ]
                    },
                }
            ],
        }
    }


@pytest.fixture
def days():
    """最後一天是今天（尚未收盤），前面幾天已收盤。"""
    today = date.today()
    return [(today - timedelta(days=n)).isoformat() for n in (4, 3, 2, 1, 0)]


# ------------------------------------------------------------------- Yahoo


def test_yahoo_parses_bars(monkeypatch, days):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: yahoo_payload(days, [10, 11, 12, 13, 14])
    )
    bars = YahooProvider(throttle=0).daily_bars("0050")
    assert [b.close for b in bars] == [10, 11, 12, 13]  # 今天的被排除
    assert bars[0].high == 11.0
    assert bars[0].low == 9.0
    assert bars == sorted(bars, key=lambda b: b.date)


def test_yahoo_excludes_todays_unfinished_bar(monkeypatch, days):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: yahoo_payload(days, [10, 11, 12, 13, 14])
    )
    bars = YahooProvider(throttle=0).daily_bars("0050")
    assert all(b.date < date.today().isoformat() for b in bars)


def test_yahoo_skips_rows_with_nulls(monkeypatch, days):
    payload = yahoo_payload(days, [10, 11, 12, 13, 14])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
    monkeypatch.setattr(D, "_http_get_json", lambda *a, **k: payload)
    bars = YahooProvider(throttle=0).daily_bars("0050")
    assert [b.close for b in bars] == [10, 12, 13]


def test_yahoo_live_price_prefers_regular_market(monkeypatch, days):
    monkeypatch.setattr(
        D,
        "_http_get_json",
        lambda *a, **k: yahoo_payload(days, [10, 11, 12, 13, 14], live=15.5),
    )
    assert YahooProvider(throttle=0).live_price("0050") == pytest.approx(15.5)


def test_yahoo_live_price_falls_back_to_previous_close(monkeypatch, days):
    monkeypatch.setattr(
        D,
        "_http_get_json",
        lambda *a, **k: yahoo_payload(days, [10, 11, 12, 13, 14], live=None),
    )
    assert YahooProvider(throttle=0).live_price("0050") == pytest.approx(14)


def test_yahoo_security_name(monkeypatch, days):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: yahoo_payload(days, [10, 11], name="富邦科技")
    )
    assert YahooProvider(throttle=0).security_name("0052") == "富邦科技"


def test_yahoo_tries_otc_suffix(monkeypatch, days):
    seen = []

    def fake(url, *a, **k):
        seen.append(url)
        if url.endswith(".TW?range=2y&interval=1d") or ".TW?" in url:
            raise DataError("not found")
        return yahoo_payload(days, [10, 11, 12])

    monkeypatch.setattr(D, "_http_get_json", fake)
    YahooProvider(throttle=0).daily_bars("6488")
    assert any(".TWO?" in u for u in seen)


def test_yahoo_respects_declared_market(monkeypatch, days):
    seen = []

    def fake(url, *a, **k):
        seen.append(url)
        return yahoo_payload(days, [10, 11, 12])

    monkeypatch.setattr(D, "_http_get_json", fake)
    YahooProvider(throttle=0, market={"0050": "listed"}).daily_bars("0050")
    assert len(seen) == 1 and ".TW?" in seen[0]


def test_yahoo_reports_all_suffix_failures(monkeypatch):
    def fake(*a, **k):
        raise DataError("boom")

    monkeypatch.setattr(D, "_http_get_json", fake)
    with pytest.raises(DataError, match="Yahoo 查無資料"):
        YahooProvider(throttle=0).daily_bars("9999")


# ----------------------------------------------------------------- FinMind


def finmind_payload(dates, closes):
    return {
        "msg": "success",
        "status": 200,
        "data": [
            {
                "date": d,
                "stock_id": "0050",
                "Trading_Volume": 1_000_000,
                "open": c - 0.5,
                "max": c + 1.0,
                "min": c - 1.0,
                "close": c,
            }
            for d, c in zip(dates, closes)
        ],
    }


def test_finmind_maps_max_min_to_high_low(monkeypatch, days):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: finmind_payload(days, [10, 11, 12, 13, 14])
    )
    bars = FinMindProvider(throttle=0).daily_bars("0050")
    assert bars[0].high == 11.0
    assert bars[0].low == 9.0
    assert [b.close for b in bars] == [10, 11, 12, 13]  # 今天排除


def test_finmind_live_price_is_last_close(monkeypatch, days):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: finmind_payload(days, [10, 11, 12, 13, 14])
    )
    assert FinMindProvider(throttle=0).live_price("0050") == pytest.approx(13)


def test_finmind_raises_on_error_status(monkeypatch):
    monkeypatch.setattr(
        D, "_http_get_json", lambda *a, **k: {"status": 402, "msg": "quota exceeded"}
    )
    with pytest.raises(DataError, match="quota exceeded"):
        FinMindProvider(throttle=0).daily_bars("0050")


def test_finmind_token_is_sent(monkeypatch, days):
    seen = {}

    def fake(url, *a, **k):
        seen["url"] = url
        return finmind_payload(days, [10, 11, 12])

    monkeypatch.setattr(D, "_http_get_json", fake)
    FinMindProvider(token="secret-token", throttle=0).daily_bars("0050")
    assert "token=secret-token" in seen["url"]


def test_finmind_empty_data_is_an_error(monkeypatch):
    monkeypatch.setattr(D, "_http_get_json", lambda *a, **k: {"status": 200, "data": []})
    with pytest.raises(DataError, match="沒有回傳日 K"):
        FinMindProvider(throttle=0).daily_bars("0050")


# ------------------------------------------------------------------- Chain


class _Stub(PriceProvider):
    def __init__(self, bars=None, price=None, name=None):
        self._bars, self._price, self._name = bars, price, name

    def daily_bars(self, ticker, months=12):
        if self._bars is None:
            raise DataError("no bars")
        return self._bars

    def live_price(self, ticker):
        if self._price is None:
            raise DataError("no price")
        return self._price

    def security_name(self, ticker):
        return self._name


def test_chain_uses_first_success():
    bars = [Bar("2026-01-01", 1, 2, 0.5, 1.5)]
    chain = ChainProvider([_Stub(), _Stub(bars=bars), _Stub(bars=[])])
    assert chain.daily_bars("0050") == bars


def test_chain_reports_every_failure():
    chain = ChainProvider([_Stub(), _Stub()])
    with pytest.raises(DataError, match="所有來源皆失敗"):
        chain.live_price("0050")


def test_chain_skips_providers_without_a_name():
    chain = ChainProvider([_Stub(name=None), _Stub(name="富邦科技")])
    assert chain.security_name("0052") == "富邦科技"


def test_chain_needs_at_least_one_provider():
    with pytest.raises(DataError):
        ChainProvider([])


# ------------------------------------------------------------- make_provider


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("yahoo", YahooProvider),
        ("finmind", FinMindProvider),
        ("twse", D.TwseProvider),
        ("auto", ChainProvider),
    ],
)
def test_make_provider(kind, expected):
    assert isinstance(make_provider(kind), expected)


def test_auto_chain_order_puts_yahoo_first():
    chain = make_provider("auto")
    assert [type(p).__name__ for p in chain.providers] == [
        "YahooProvider",
        "TwseProvider",
        "FinMindProvider",
    ]


def test_unknown_provider_lists_the_valid_ones():
    with pytest.raises(DataError, match="yahoo"):
        make_provider("bloomberg")
