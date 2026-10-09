"""分析记录持久化与未来收益复核测试。"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stock_analysis import storage


def _price_frame(periods: int = 40, start: str = "2024-01-01", base: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    close = base + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [1_000_000] * periods,
        }
    )


def _record(analysis_date: str = "2024-01-05", close: float = 104.0, symbol: str = "TEST") -> dict:
    return {
        "symbol": symbol,
        "analysis_date": analysis_date,
        "close_price": close,
        "indicators": {"close": close},
    }


def test_safe_name():
    assert storage.safe_name("GC=F") == "GC_F"
    assert storage.safe_name("A/B:C") == "A_B_C"


def test_save_and_load_analysis(tmp_path):
    record = _record()
    record["markdown"] = "# TEST 报告"
    paths = storage.save_analysis(record, str(tmp_path))

    assert paths["json"].endswith(".json")
    assert paths["markdown"].endswith(".md")

    with open(paths["markdown"], encoding="utf-8") as handle:
        assert handle.read() == "# TEST 报告"

    loaded = storage.load_record(paths["json"])
    assert loaded["symbol"] == "TEST"
    assert loaded["record_version"] == storage.RECORD_VERSION
    assert loaded["forward_returns"] is None  # 预留字段
    assert "saved_at_utc" in loaded

    assert storage.iter_record_paths(str(tmp_path)) == [paths["json"]]


def test_evaluate_forward_returns_uses_only_future_bars():
    """未来收益只能使用分析日之后的 K 线，且基准价取分析日收盘价。"""
    frame = _price_frame(periods=30)
    result = storage.evaluate_forward_returns(_record(analysis_date="2024-01-05"), frame)

    assert result["base_close"] == pytest.approx(104.0)
    assert result["basis"] == "复核数据中分析日收盘价"

    five_day = result["horizons"]["5d"]
    assert five_day["date"] == "2024-01-12"  # 分析日之后的第 5 个交易日
    assert five_day["return"] == pytest.approx(round(109 / 104 - 1, 4))

    twenty_day = result["horizons"]["20d"]
    assert twenty_day["return"] is not None
    assert result["complete"] is True

    # MFE/MAE 只考虑分析日之后的 20 根 K 线（后续收盘价 105..124，最高 124.5）
    assert result["max_favorable_excursion"] == pytest.approx(round((124.5) / 104 - 1, 4))
    assert result["max_adverse_excursion"] == pytest.approx(round((104.5) / 104 - 1, 4))
    assert result["future_bars"] == 25  # 2024-01-08 起的交易日数量


def test_evaluate_forward_returns_incomplete_window():
    frame = _price_frame(periods=12, start="2024-01-01")
    result = storage.evaluate_forward_returns(_record(analysis_date="2024-01-05"), frame)

    assert result["complete"] is False
    assert result["horizons"]["5d"]["return"] is not None
    assert result["horizons"]["10d"]["return"] is None
    assert "不足" in result["horizons"]["10d"]["note"]
    assert "note" in result


def test_evaluate_forward_returns_no_future_data():
    frame = _price_frame(periods=5, start="2024-01-01")  # 分析日即最后一根
    result = storage.evaluate_forward_returns(_record(analysis_date="2024-01-05"), frame)
    assert result["future_bars"] == 0
    assert "error" in result


def test_evaluate_forward_returns_empty_frame():
    result = storage.evaluate_forward_returns(_record(), pd.DataFrame())
    assert "error" in result


def test_review_records_updates_history(tmp_path):
    record = _record(analysis_date="2024-01-05")
    record["markdown"] = "# TEST"
    paths = storage.save_analysis(record, str(tmp_path))

    frame = _price_frame(periods=40)
    updates = storage.review_records(
        str(tmp_path),
        min_age_days=30,
        fetch_fn=lambda symbol: frame,
        reference_date=datetime(2024, 3, 1),
    )

    assert len(updates) == 1
    saved = storage.load_record(paths["json"])
    assert saved["forward_returns"]["complete"] is True
    assert saved["forward_returns"]["horizons"]["20d"]["return"] is not None


def test_review_records_skips_recent_and_completed(tmp_path):
    storage.save_analysis(_record(analysis_date="2024-02-28"), str(tmp_path))
    fetched = []

    updates = storage.review_records(
        str(tmp_path),
        min_age_days=30,
        fetch_fn=lambda symbol: fetched.append(symbol) or _price_frame(),
        reference_date=datetime(2024, 3, 1),
    )
    assert updates == []
    assert fetched == []  # 未满观察期，不触发取数

    # 已完成的记录默认跳过
    storage.save_analysis(_record(analysis_date="2024-01-05"), str(tmp_path))
    storage.review_records(
        str(tmp_path),
        min_age_days=30,
        fetch_fn=lambda symbol: _price_frame(periods=40),
        reference_date=datetime(2024, 3, 1),
    )
    second_pass = storage.review_records(
        str(tmp_path),
        min_age_days=30,
        fetch_fn=lambda symbol: fetched.append(symbol) or _price_frame(periods=40),
        reference_date=datetime(2024, 3, 1),
    )
    assert second_pass == []
    assert fetched == []


def test_review_records_fetch_failure_is_tolerated(tmp_path):
    storage.save_analysis(_record(analysis_date="2024-01-05"), str(tmp_path))

    def boom(symbol):
        raise RuntimeError("network down")

    updates = storage.review_records(
        str(tmp_path), min_age_days=30, fetch_fn=boom, reference_date=datetime(2024, 3, 1)
    )
    assert updates == []


def test_review_records_skips_invalid_json(tmp_path):
    (tmp_path / "TEST").mkdir(parents=True, exist_ok=True)
    (tmp_path / "TEST" / "broken.json").write_text("{ not json", encoding="utf-8")
    updates = storage.review_records(
        str(tmp_path), min_age_days=0, fetch_fn=lambda s: _price_frame(), force=True
    )
    assert updates == []


def test_forward_returns_serializable(tmp_path):
    record = _record()
    record["forward_returns"] = storage.evaluate_forward_returns(record, _price_frame())
    paths = storage.save_analysis(record, str(tmp_path))
    with open(paths["json"], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["forward_returns"]["horizons"]["5d"]["return"] is not None
