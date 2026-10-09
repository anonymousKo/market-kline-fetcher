"""关键价位与价格结构测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis import key_levels


def _from_closes(closes, start="2024-01-01"):
    dates = pd.bdate_range(start=start, periods=len(closes))
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close.shift(1).fillna(close),
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [1_000_000] * len(closes),
        }
    )


def _zigzag():
    segments = [
        (100.0, 88.0, 8),
        (88.0, 112.0, 8),
        (112.0, 90.0, 8),
        (90.0, 108.0, 8),
        (108.0, 92.0, 8),
        (92.0, 104.0, 8),
        (104.0, 95.0, 8),
        (95.0, 101.0, 8),
    ]
    closes = []
    for start, end, count in segments:
        closes.extend(np.linspace(start, end, count, endpoint=False).tolist())
    closes.append(101.0)
    return _from_closes(closes)


def test_find_pivots_detects_turning_points():
    df = _zigzag()
    highs, lows = key_levels.find_pivots(df, span=2)
    assert len(highs) >= 3
    assert len(lows) >= 3
    assert all(item["price"] is not None and item["date"] for item in highs + lows)


def test_find_pivots_excludes_unconfirmed_recent_bars():
    df = _zigzag()
    span = 2
    highs, lows = key_levels.find_pivots(df, span=span)
    recent_dates = {key_levels._date(d) for d in df["Date"].tail(span)}
    for pivot in highs + lows:
        assert pivot["date"] not in recent_dates, "最近 span 根 K 线内的枢轴尚未确认，不应使用"


def test_find_pivots_insufficient_data():
    df = _from_closes([1.0, 2.0, 3.0])
    highs, lows = key_levels.find_pivots(df, span=2)
    assert highs == [] and lows == []


def test_cluster_zones_merges_and_separates():
    pivots = [
        {"date": "2024-01-02", "price": 100.0},
        {"date": "2024-01-10", "price": 100.5},
        {"date": "2024-02-01", "price": 120.0},
    ]
    merged = key_levels.cluster_zones(pivots, tolerance=2.0)
    sizes = sorted(zone["touches"] for zone in merged)
    assert sizes == [1, 2]

    split = key_levels.cluster_zones(pivots, tolerance=0.1)
    assert len(split) == 3


def test_cluster_zones_zero_tolerance():
    pivots = [{"date": "2024-01-02", "price": 100.0}]
    assert key_levels.cluster_zones(pivots, tolerance=0.0) == []


def test_build_key_levels_structure_and_ordering():
    result = key_levels.build_key_levels(_zigzag())

    assert result["available"] is True
    assert result["as_of"] is not None
    assert "recent_extremes" in result
    assert "last20d" in result["recent_extremes"]
    assert result["insufficient_evidence"] is False

    close = result["close"]
    for zone in result["support_zones"]:
        assert zone["level"] < close
        assert zone["range"][0] <= zone["range"][1]
    for zone in result["resistance_zones"]:
        assert zone["level"] > close

    support_levels = [zone["level"] for zone in result["support_zones"]]
    assert support_levels == sorted(support_levels, reverse=True)  # 由近及远
    resistance_levels = [zone["level"] for zone in result["resistance_zones"]]
    assert resistance_levels == sorted(resistance_levels)

    assert result["nearest_support"]["level"] == support_levels[0]
    assert result["nearest_resistance"]["level"] == resistance_levels[0]
    assert "cluster_rule" in result
    assert set(result["ma_distances_pct"]).issuperset({"ma20", "ma50", "ma200"})


def test_build_key_levels_insufficient_evidence():
    df = _from_closes([100.0, 101.0, 102.0, 101.5, 102.5, 103.0])
    result = key_levels.build_key_levels(df, pivot_span=5)
    assert result["insufficient_evidence"] is True
    assert result["support_zones"] == [] and result["resistance_zones"] == []
    assert any("暂无足够证据" in note for note in result["notes"])


def test_build_key_levels_empty():
    result = key_levels.build_key_levels(pd.DataFrame())
    assert result["available"] is False
    assert result["insufficient_evidence"] is True


def test_breakout_note_when_price_above_prior_high():
    closes = [100.0] * 30
    closes[20] = 130.0  # 前 20 日高点（剔除最近 5 日）
    df = _from_closes(closes)
    # 重新构造：前 25~6 根区间内的最高点 + 之后突破
    closes = [100.0] * 20 + [110.0] * 4 + [105.0, 106.0, 107.0, 108.0, 109.0, 115.0]
    df = _from_closes(closes)
    result = key_levels.build_key_levels(df)
    assert any("已站上前 20 日高点" in note for note in result["breakout_notes"])
