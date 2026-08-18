"""網頁後端 GridService 的測試。

重點在於：頁面拿到的 JSON 是否忠實反映引擎的決策，以及回填成交會不會
把狀態寫壞。HTTP 層只是薄薄一層轉接，所以直接測 service。
"""

import json
from datetime import date

import pytest
import yaml
from conftest import bars_with_atr

from atrgrid.data import export_csv
from atrgrid.engine import Decision
from atrgrid.state import load_state
from atrgrid.web import ApiError, GridService, decision_to_dict


@pytest.fixture
def workspace(tmp_path):
    """一個完整的迷你工作區：設定檔 + 日 K + 狀態檔。"""
    config = tmp_path / "config"
    config.mkdir()
    bars_dir = tmp_path / "bars"

    # 最後一根 K 棒收在昨天，價格 100，ATR 約 2.0
    bars = bars_with_atr(80, close=100.0, spread=2.0,
                         end_date=date.today().isoformat())
    export_csv(bars, bars_dir / "0052.csv")
    export_csv(bars, bars_dir / "00725B.csv")

    grid = {
        "atr_period": 14, "atr_multiplier": 0.5, "min_step_pct": 0.1,
        "max_step_pct": 20.0, "max_buy_rungs": 5, "max_sell_rungs": 5,
        "max_rungs_per_day": 2, "gap_atr_limit": 3.0, "drift_mode": "off",
        "drift_beta": 0.0, "trend_ema_period": 60, "allow_loss_sell": False,
    }
    (config / "settings.yaml").write_text(yaml.safe_dump({
        "fees": {"discount": "0.28", "minimum": 1},
        "risk": {"cash": 200000, "cash_floor": 0},
        "defaults": {"equity": grid, "bond": grid, "leveraged": grid, "stock": grid},
    }), encoding="utf-8")

    (config / "portfolio.yaml").write_text(yaml.safe_dump({
        "holdings": [
            {"ticker": "0052", "name": "富邦科技", "class": "equity",
             "shares": 1000, "avg_cost": 50.0, "ticker_verified": True},
            {"ticker": "00725B", "name": "國泰投資級公司債", "class": "bond",
             "shares": 2000, "avg_cost": 30.0, "ticker_verified": False},
        ]
    }), encoding="utf-8")

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "version": 2, "cash": 200000.0,
        "positions": {
            "0052": {"shares": 1000, "anchor": 100.0, "rung": 0,
                     "baseline_shares": 1000, "lots": [
                         {"date": "2026-01-01", "price": 50.0,
                          "shares": 1000, "source": "initial"}]},
            "00725B": {"shares": 2000, "anchor": 100.0, "rung": 0,
                       "baseline_shares": 2000, "lots": [
                           {"date": "2026-01-01", "price": 30.0,
                            "shares": 2000, "source": "initial"}]},
        },
        "trades": [],
    }), encoding="utf-8")

    return GridService(config_dir=config, state_path=state,
                       provider_kind="csv", csv_dir=bars_dir)


# ------------------------------------------------------------------ snapshot


def test_snapshot_reports_holdings_and_lots(workspace):
    snap = workspace.snapshot()
    assert snap["cash"] == 200000
    assert len(snap["holdings"]) == 2
    fubon = next(h for h in snap["holdings"] if h["ticker"] == "0052")
    assert fubon["shares"] == 1000
    assert fubon["lotShares"] == 50       # 50 × 100 = 5000 < 5012.53
    assert fubon["anchor"] == 100.0


def test_snapshot_counts_unverified_tickers(workspace):
    assert workspace.snapshot()["unverified"] == 1


def test_snapshot_does_not_need_the_network(workspace):
    workspace.provider_kind = "yahoo"   # 就算設成線上來源
    snap = workspace.snapshot()          # snapshot 也不該連網
    assert len(snap["holdings"]) == 2


# -------------------------------------------------------------------- quotes


def test_quotes_returns_prices_for_every_holding(workspace):
    res = workspace.quotes()
    assert set(res["prices"]) == {"0052", "00725B"}
    assert res["errors"] == {}


def test_quotes_isolates_per_ticker_failures(workspace, tmp_path):
    (workspace.csv_dir / "00725B.csv").unlink()
    res = workspace.quotes()
    assert "0052" in res["prices"]
    assert "00725B" in res["errors"]


# -------------------------------------------------------------------- advise


def test_advise_uses_supplied_prices(workspace):
    res = workspace.advise(prices={"0052": 98.9})
    buy = next(d for d in res["decisions"] if d["ticker"] == "0052")
    assert buy["action"] == "BUY"
    assert buy["price"] == 98.9
    assert buy["rungs"] == 1


def test_advise_skips_unverified_ticker(workspace):
    res = workspace.advise(prices={"00725B": 90.0})
    bond = next(d for d in res["decisions"] if d["ticker"] == "00725B")
    assert bond["action"] == "SKIP"
    assert any("代號" in b for b in bond["blocks"])


def test_advise_summary_totals_match_decisions(workspace):
    res = workspace.advise(prices={"0052": 97.5})
    acts = [d for d in res["decisions"] if d["shares"] > 0]
    assert res["summary"]["orders"] == sum(d["rungs"] for d in acts)
    assert res["summary"]["cost"] == sum(d["fee"] + d["tax"] for d in acts)


def test_advise_reports_multi_rung_as_split_orders(workspace):
    res = workspace.advise(prices={"0052": 97.5})
    buy = next(d for d in res["decisions"] if d["ticker"] == "0052")
    assert buy["rungs"] == 2
    assert buy["fee"] == 2          # 拆兩筆，不是合併的 3 元
    assert buy["shares"] == buy["rungs"] * buy["lotShares"]


def test_advise_does_not_mutate_saved_state(workspace):
    before = workspace.state_path.read_text(encoding="utf-8")
    workspace.advise(prices={"0052": 97.5})
    assert workspace.state_path.read_text(encoding="utf-8") == before


def test_advise_marks_missing_data_as_skip(workspace):
    (workspace.csv_dir / "0052.csv").unlink()
    res = workspace.advise()
    bad = next(d for d in res["decisions"] if d["ticker"] == "0052")
    assert bad["action"] == "SKIP"
    assert any("資料取得失敗" in b for b in bad["blocks"])


# -------------------------------------------------------------------- record


def test_record_buy_updates_state_on_disk(workspace):
    res = workspace.record({"ticker": "0052", "action": "BUY",
                            "shares": 50, "price": 98.9, "rungs": 1,
                            "step": 1.0})
    assert res["ok"] is True
    assert res["rung"] == 1
    assert res["shares"] == 1050

    saved = load_state(workspace.state_path)
    position = saved.positions["0052"]
    assert position.shares == 1050
    assert position.anchor == pytest.approx(99.0)   # 100 − 1.0 × 1
    assert saved.cash < 200000
    assert len(saved.trades) == 1


def test_record_sell_realizes_profit(workspace):
    res = workspace.record({"ticker": "0052", "action": "SELL",
                            "shares": 50, "price": 101.0, "rungs": 1,
                            "step": 1.0})
    assert res["realizedPnl"] > 0        # 成本 50，賣在 101
    assert res["tax"] > 0                # 股票型 ETF 要課證交稅
    saved = load_state(workspace.state_path)
    assert saved.positions["0052"].shares == 950


def test_record_bond_sale_is_tax_free(workspace):
    res = workspace.record({"ticker": "00725B", "action": "SELL",
                            "shares": 50, "price": 101.0})
    assert res["tax"] == 0


def test_record_rejects_overselling(workspace):
    with pytest.raises(ApiError, match="超過持股"):
        workspace.record({"ticker": "0052", "action": "SELL",
                          "shares": 99999, "price": 100.0})


def test_record_rejects_unknown_ticker(workspace):
    with pytest.raises(ApiError, match="不在狀態檔中"):
        workspace.record({"ticker": "9999", "action": "BUY",
                          "shares": 1, "price": 10.0})


def test_record_rejects_bad_action(workspace):
    with pytest.raises(ApiError, match="action"):
        workspace.record({"ticker": "0052", "action": "HOLD",
                          "shares": 1, "price": 10.0})


@pytest.mark.parametrize("bad", [
    {"shares": 0, "price": 10.0},
    {"shares": -5, "price": 10.0},
    {"shares": 10, "price": 0},
    {"shares": "abc", "price": 10.0},
])
def test_record_rejects_bad_numbers(workspace, bad):
    with pytest.raises(ApiError):
        workspace.record({"ticker": "0052", "action": "BUY", **bad})


def test_record_without_step_snaps_anchor_to_fill_price(workspace):
    workspace.record({"ticker": "0052", "action": "BUY",
                      "shares": 50, "price": 97.3})
    assert load_state(workspace.state_path).positions["0052"].anchor == \
        pytest.approx(97.3)


def test_failed_record_leaves_state_untouched(workspace):
    before = workspace.state_path.read_text(encoding="utf-8")
    with pytest.raises(ApiError):
        workspace.record({"ticker": "0052", "action": "SELL",
                          "shares": 99999, "price": 100.0})
    assert workspace.state_path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------- cash


def test_set_cash_persists(workspace):
    workspace.set_cash(12345)
    assert load_state(workspace.state_path).cash == pytest.approx(12345)


# --------------------------------------------------------------- 序列化格式


def test_decision_to_dict_is_json_serialisable():
    payload = decision_to_dict(
        Decision(ticker="0052", name="富邦科技", asset_class="equity",
                 action="BUY", shares=83, rungs=1, lot_shares=83, price=60.35)
    )
    json.dumps(payload)          # 不該拋出
    assert payload["lotShares"] == 83
    assert payload["realizedPnl"] is None
