"""技术指标计算测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis import indicators


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------


def test_sma_basic():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = indicators.sma(series, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_uses_adjust_false():
    series = pd.Series([1.0, 2.0, 3.0])
    out = indicators.ema(series, 2)
    alpha = 2 / 3
    expected_1 = out.iloc[0] * (1 - alpha) + 2.0 * alpha
    assert out.iloc[1] == pytest.approx(expected_1)


def test_wilder_smooth_seed_and_recursion():
    series = pd.Series(np.arange(1, 21, dtype=float))
    out = indicators.wilder_smooth(series, 5)

    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(3.0)  # mean(1..5)

    prev = 3.0
    for i in range(5, len(series)):
        prev = (prev * 4 + series.iloc[i]) / 5
        assert out.iloc[i] == pytest.approx(prev)

    # 数据不足时全为 NaN
    short = indicators.wilder_smooth(pd.Series([1.0, 2.0]), 5)
    assert short.isna().all()


def test_wilder_smooth_skips_leading_nan():
    series = pd.Series([np.nan] + list(np.arange(1, 21, dtype=float)))
    out = indicators.wilder_smooth(series, 5)
    assert np.isnan(out.iloc[0])
    assert out.iloc[5] == pytest.approx(3.0)  # mean(1..5)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def _naive_rsi(values, length=14):
    """独立的参考实现（用于交叉验证向量化实现）。"""
    gains, losses = [], []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    result = [np.nan] * len(values)
    if len(gains) < length:
        return result

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length
    result[length] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i - 1]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i - 1]) / length
        result[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def test_rsi_matches_reference_implementation():
    rng = np.random.default_rng(3)
    values = 100 + np.cumsum(rng.normal(0, 1, 120))
    close = pd.Series(values)
    ours = indicators.rsi(close, 14)
    reference = _naive_rsi(list(values), 14)

    for index in range(14, len(values)):
        assert ours.iloc[index] == pytest.approx(reference[index], rel=1e-9), index


def test_rsi_extremes_and_warmup():
    up = indicators.rsi(pd.Series(np.arange(1, 41, dtype=float)), 14)
    assert up.iloc[-1] == pytest.approx(100.0)
    assert up.iloc[:14].isna().all()  # 预热期为 NaN

    down = indicators.rsi(pd.Series(np.arange(40, 0, -1, dtype=float)), 14)
    assert down.iloc[-1] == pytest.approx(0.0)

    flat = indicators.rsi(pd.Series([5.0] * 40), 14)
    assert flat.iloc[-1] == pytest.approx(50.0)


def test_rsi_bounds(sample_daily):
    rsi = indicators.rsi(sample_daily["Close"], 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_definition():
    close = pd.Series(np.linspace(10, 50, 200))
    out = indicators.macd(close, 12, 26, 9)
    expected_dif = indicators.ema(close, 12) - indicators.ema(close, 26)
    assert np.allclose(out["dif"], expected_dif)
    assert np.allclose(out["dea"], indicators.ema(expected_dif, 9))
    assert np.allclose(out["hist"], out["dif"] - out["dea"])  # 柱状图 = DIF - DEA


def test_macd_no_lookahead(sample_daily):
    """指标不得使用未来数据：截断序列后，历史值必须保持不变。"""
    close = sample_daily["Close"]
    full = indicators.macd(close)
    truncated = indicators.macd(close.iloc[:200])
    assert full["hist"].iloc[199] == pytest.approx(truncated["hist"].iloc[199])
    assert full["dif"].iloc[150] == pytest.approx(indicators.macd(close.iloc[:151])["dif"].iloc[150])


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def test_true_range_formula():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([9.0, 10.0, 10.0])
    close = pd.Series([9.5, 11.0, 10.5])
    tr = indicators.true_range(high, low, close)
    # 第一根 K 线无前收盘价 -> 取 High - Low
    assert tr.iloc[0] == pytest.approx(1.0)
    assert tr.iloc[1] == pytest.approx(max(12 - 10, abs(12 - 9.5), abs(10 - 9.5)))
    assert tr.iloc[2] == pytest.approx(max(11 - 10, abs(11 - 11.0), abs(10 - 11.0)))


def test_atr_constant_range():
    high = pd.Series([10.0] * 30)
    low = pd.Series([9.0] * 30)
    close = pd.Series([9.5] * 30)
    out = indicators.atr(high, low, close, 14)
    assert out.iloc[-1] == pytest.approx(1.0)
    # 前 13 个位置为预热期，第 14 根 K 线起有值
    assert out.iloc[:13].isna().all()
    assert out.iloc[13] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 量能与收益率
# ---------------------------------------------------------------------------


def test_volume_stats():
    volume = pd.Series([100.0] * 19 + [200.0])
    out = indicators.volume_stats(volume, 20)
    assert out["volume_avg"].iloc[-1] == pytest.approx(105.0)
    assert out["volume_ratio"].iloc[-1] == pytest.approx(200 / 105)


def test_rolling_return_windows():
    close = pd.Series(np.arange(1, 200, dtype=float))
    out = indicators.rolling_return(close, 21)
    assert out.iloc[-1] == pytest.approx(close.iloc[-1] / close.iloc[-22] - 1)


def test_add_indicators_ma200_and_no_lookahead(sample_daily):
    data = indicators.add_indicators(sample_daily)
    expected = sample_daily["Close"].iloc[-200:].mean()
    assert data["MA200"].iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert data["MA200"].iloc[:199].isna().all()
    # 截断后历史值不变（无未来数据）
    truncated = indicators.add_indicators(sample_daily.iloc[:250])
    assert data["MA50"].iloc[249] == pytest.approx(truncated["MA50"].iloc[249])


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------


def test_daily_snapshot_full_history(sample_daily):
    snapshot = indicators.daily_indicator_snapshot(sample_daily)
    assert snapshot["available"] is True
    assert snapshot["ma"]["ma200"] is not None
    assert snapshot["rsi"]["rsi14"] is not None
    assert snapshot["returns_pct"]["6m"]["value"] is not None
    assert snapshot["atr"]["atr14"] is not None
    assert snapshot["volume"]["volume_ratio20"] is not None
    assert snapshot["trend"]["trend"] in {"上升", "下降", "震荡"}


def test_daily_snapshot_reports_insufficient_history(sample_daily):
    short = sample_daily.tail(60).reset_index(drop=True)
    snapshot = indicators.daily_indicator_snapshot(short)
    assert snapshot["ma"]["ma200"] is None
    assert snapshot["returns_pct"]["6m"]["value"] is None
    joined = " ".join(snapshot["warnings"])
    assert "MA200" in joined
    assert "6m" in joined


def test_daily_snapshot_empty():
    snapshot = indicators.daily_indicator_snapshot(pd.DataFrame())
    assert snapshot["available"] is False


# ---------------------------------------------------------------------------
# 相对强弱
# ---------------------------------------------------------------------------


def test_relative_performance_windows():
    asset = pd.Series(np.linspace(100, 110, 64))
    bench = pd.Series(np.linspace(100, 105, 64))
    out = indicators.relative_performance(asset, bench, windows={"1m": 21})
    window = out["windows"]["1m"]
    assert window["asset"] == pytest.approx(round((asset.iloc[-1] / asset.iloc[-22] - 1) * 100, 2))
    assert window["benchmark"] == pytest.approx(
        round((bench.iloc[-1] / bench.iloc[-22] - 1) * 100, 2)
    )
    assert window["excess"] == pytest.approx(round(window["asset"] - window["benchmark"], 2))
    assert out["available"] is True


def test_relative_performance_insufficient_data():
    out = indicators.relative_performance(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    assert out["available"] is False
    assert out["warnings"]


def test_classify_trend_rules():
    up, _ = indicators.classify_trend(110.0, 105.0, 100.0, True, "MA20", "MA50")
    assert up == "上升"
    down, _ = indicators.classify_trend(90.0, 95.0, 100.0, False, "MA20", "MA50")
    assert down == "下降"
    sideways, _ = indicators.classify_trend(100.0, 105.0, 100.0, True, "MA20", "MA50")
    assert sideways == "震荡"
    unknown, evidence = indicators.classify_trend(100.0, None, None, None, "MA20", "MA50")
    assert unknown == "数据不足" and evidence
