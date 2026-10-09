"""数据规范化、清洗与质量检查测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis import kline


def test_normalize_multiindex_columns_and_datetime_index(sample_daily):
    """模拟 yfinance 单标的返回的 MultiIndex 列 + DatetimeIndex。"""
    raw = sample_daily.set_index("Date")
    raw.columns = pd.MultiIndex.from_product([list(raw.columns), ["TEST"]])

    out = kline.normalize_ohlcv(raw)
    assert list(out.columns) == kline.STANDARD_COLUMNS
    assert out["Date"].is_monotonic_increasing
    assert len(out) == len(sample_daily)


def test_normalize_renames_aliases_and_parses_string_dates(sample_daily):
    raw = sample_daily.copy()
    raw.columns = ["date", "open", "high", "low", "close", "volume"]
    raw["date"] = raw["date"].dt.strftime("%Y/%m/%d")  # 字符串日期

    out = kline.normalize_ohlcv(raw)
    assert list(out.columns) == kline.STANDARD_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_normalize_sorts_and_dedupes():
    rows = [
        ("2024-01-03", 3.0, 3.5, 2.5, 3.2, 10.0),
        ("2024-01-01", 1.0, 1.5, 0.5, 1.2, 10.0),
        ("2024-01-01", 1.1, 1.6, 0.6, 1.3, 11.0),  # 重复日期
        ("2024-01-02", 2.0, 2.5, 1.5, 2.2, 10.0),
    ]
    raw = pd.DataFrame(rows, columns=kline.STANDARD_COLUMNS)
    out = kline.normalize_ohlcv(raw)

    assert out["Date"].is_monotonic_increasing
    assert len(out) == 3  # 去重后
    # 保留最后一条（修正后的数据）
    first = out.loc[out["Date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert first["Open"] == pytest.approx(1.1)
    notes = " ".join(out.attrs.get("notes", []))
    assert "排序" in notes and "重复" in notes


def test_normalize_missing_required_column_raises():
    raw = pd.DataFrame({"Date": ["2024-01-01"], "Open": [1.0], "High": [1.5]})
    with pytest.raises(ValueError):
        kline.normalize_ohlcv(raw)  # 缺少 Close

    with pytest.raises(ValueError):
        kline.normalize_ohlcv(pd.DataFrame({"Open": [1.0], "Close": [1.0]}))  # 缺少 Date


def test_drop_invalid_rows():
    rows = [
        ("2024-01-01", 0.0, 1.0, 0.0, 1.0, 100.0),  # 非正价格
        ("2024-01-02", 2.0, 1.5, 2.5, 2.0, 100.0),  # High < Low
        ("2024-01-03", 3.0, 3.5, 2.5, 3.2, -5.0),  # 负成交量
        ("2024-01-04", np.nan, np.nan, np.nan, np.nan, 100.0),  # 价格全缺失
        ("2024-01-05", 5.0, 5.5, 4.5, 5.2, 100.0),  # 正常
    ]
    raw = pd.DataFrame(rows, columns=kline.STANDARD_COLUMNS)
    normalized = kline.normalize_ohlcv(raw)
    cleaned, notes = kline.drop_invalid_rows(normalized)

    assert len(cleaned) == 2  # 只保留 01-03（成交量置空）与 01-05
    assert cleaned["Date"].iloc[0] == pd.Timestamp("2024-01-03")
    assert np.isnan(cleaned["Volume"].iloc[0])  # 负成交量 -> NaN，不伪造
    assert cleaned["Volume"].iloc[1] == pytest.approx(100.0)
    joined = " ".join(notes)
    assert "非正价格" in joined
    assert "High < Low" in joined
    assert "成交量为负" in joined


def test_check_quality_insufficient_history(sample_daily):
    quality = kline.check_quality(sample_daily.tail(60).reset_index(drop=True), "TEST", min_bars=260)
    assert quality.ok is False
    assert any("历史数据不足" in issue for issue in quality.issues)
    assert quality.bars == 60


def test_check_quality_detects_nan_prices(sample_daily):
    dirty = sample_daily.copy()
    dirty.loc[5, "Close"] = np.nan
    quality = kline.check_quality(dirty, "TEST", min_bars=100)
    assert any("Close 存在" in issue for issue in quality.issues)


def test_check_quality_stale_and_incomplete_bar(sample_daily):
    last_date = pd.Timestamp(sample_daily["Date"].iloc[-1])

    stale = kline.check_quality(
        sample_daily, "TEST", min_bars=100, as_of=last_date + pd.Timedelta(days=30)
    )
    assert any("不新鲜" in w for w in stale.warnings)

    same_day = kline.check_quality(sample_daily, "TEST", min_bars=100, as_of=last_date)
    assert same_day.incomplete_bar is True
    assert any("尚未收盘" in w for w in same_day.warnings)


def test_check_quality_empty():
    quality = kline.check_quality(pd.DataFrame(columns=kline.STANDARD_COLUMNS), "TEST")
    assert quality.ok is False
    assert any("未获取到" in issue for issue in quality.issues)


def test_prepare_daily_end_to_end(sample_daily):
    daily, quality = kline.prepare_daily(sample_daily, "TEST", min_bars=260)
    assert len(daily) == len(sample_daily)
    assert quality.bars == len(sample_daily)
    assert quality.adjusted is True
    assert "auto_adjust" in quality.adjustment_note


def test_infer_market():
    assert kline.infer_market("MSFT") == ("us_equity", "USD")
    assert kline.infer_market("GC=F") == ("futures", "USD")
    assert kline.infer_market("000001.SZ") == ("china_a_shenzhen", "CNY")
    assert kline.infer_market("600000.SS") == ("china_a_shanghai", "CNY")
    assert kline.infer_market("0700.HK") == ("hong_kong", "HKD")
    assert kline.infer_market("SPY")[0] == "us_etf"
