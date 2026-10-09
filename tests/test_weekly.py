"""日线聚合周线的正确性测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis import indicators, kline


def _df(rows):
    return pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])


WEEK1 = [
    ("2024-01-01", 10.0, 12.0, 9.0, 11.0, 100.0),
    ("2024-01-02", 11.0, 13.0, 10.0, 12.0, 200.0),
    ("2024-01-03", 12.0, 14.0, 11.0, 13.0, 300.0),
    ("2024-01-04", 13.0, 15.0, 12.0, 14.0, 400.0),
    ("2024-01-05", 14.0, 16.0, 13.0, 15.0, 500.0),
]

WEEK2 = [
    ("2024-01-08", 15.0, 17.0, 14.0, 16.0, 100.0),
    ("2024-01-09", 16.0, 18.0, 15.0, 17.0, 100.0),
    ("2024-01-10", 17.0, 19.0, 16.0, 18.0, 100.0),
    ("2024-01-11", 18.0, 20.0, 17.0, 19.0, 100.0),
    ("2024-01-12", 19.0, 21.0, 18.0, 20.0, 100.0),
]

WEEK3 = [
    ("2024-01-15", 20.0, 22.0, 19.0, 21.0, 50.0),
    ("2024-01-16", 21.0, 23.0, 20.0, 22.0, 60.0),
]


def _daily(*weeks):
    rows = []
    for week in weeks:
        rows.extend(week)
    df = _df(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_aggregate_weekly_ohlcv_rules():
    weekly, complete, warnings = kline.aggregate_weekly(
        _daily(WEEK1, WEEK2), as_of=pd.Timestamp("2024-01-12")
    )
    assert len(weekly) == 2
    assert complete is True
    assert not warnings

    first = weekly.iloc[0]
    assert first["WeekStart"] == pd.Timestamp("2024-01-01")
    assert first["WeekEnd"] == pd.Timestamp("2024-01-05")
    assert first["Date"] == pd.Timestamp("2024-01-05")  # 以该周最后一个交易日代表
    assert first["Open"] == pytest.approx(10.0)  # 第一根有效日 K 的开盘价
    assert first["High"] == pytest.approx(16.0)  # 全周最高
    assert first["Low"] == pytest.approx(9.0)  # 全周最低
    assert first["Close"] == pytest.approx(15.0)  # 最后一根有效日 K 的收盘价
    assert first["Volume"] == pytest.approx(1500.0)  # 成交量之和
    assert first["Bars"] == 5

    second = weekly.iloc[1]
    assert second["Open"] == pytest.approx(15.0)
    assert second["High"] == pytest.approx(21.0)
    assert second["Low"] == pytest.approx(14.0)
    assert second["Close"] == pytest.approx(20.0)
    assert second["Volume"] == pytest.approx(500.0)


def test_incomplete_week_is_excluded():
    weekly, complete, warnings = kline.aggregate_weekly(
        _daily(WEEK1, WEEK2, WEEK3), as_of=pd.Timestamp("2024-01-16")
    )
    assert complete is False
    assert len(weekly) == 2  # 进行中的第 3 周被剔除
    assert weekly["WeekEnd"].iloc[-1] == pd.Timestamp("2024-01-12")
    assert any("尚未结束" in w for w in warnings)


def test_incomplete_week_can_be_kept():
    weekly, complete, _ = kline.aggregate_weekly(
        _daily(WEEK1, WEEK2, WEEK3), as_of=pd.Timestamp("2024-01-16"), drop_incomplete=False
    )
    assert complete is False
    assert len(weekly) == 3
    assert weekly.iloc[-1]["Bars"] == 2


def test_week_becomes_complete_in_later_iso_week():
    # 该周最后一根日 K 是周二，但参考日期已进入下一 ISO 周 -> 视为已完成
    weekly, complete, warnings = kline.aggregate_weekly(
        _daily(WEEK1, WEEK2, WEEK3), as_of=pd.Timestamp("2024-01-22")
    )
    assert complete is True
    assert len(weekly) == 3
    assert weekly.iloc[-1]["WeekEnd"] == pd.Timestamp("2024-01-16")
    assert not any("尚未结束" in w for w in warnings)


def test_holiday_shortened_week_completes():
    # 周五为节假日，只有周一到周四 4 根日 K
    holiday_week = [
        ("2024-02-12", 1.0, 2.0, 0.5, 1.5, 10.0),
        ("2024-02-13", 1.5, 2.5, 1.0, 2.0, 10.0),
        ("2024-02-14", 2.0, 3.0, 1.5, 2.5, 10.0),
        ("2024-02-15", 2.5, 3.5, 2.0, 3.0, 10.0),
    ]
    weekly, complete, _ = kline.aggregate_weekly(_daily(holiday_week), as_of=pd.Timestamp("2024-02-20"))
    assert complete is True
    assert len(weekly) == 1
    assert weekly.iloc[0]["Bars"] == 4
    assert weekly.iloc[0]["Volume"] == pytest.approx(40.0)


def test_first_valid_open_and_last_valid_close():
    week = [
        ("2024-03-04", float("nan"), 12.0, 9.0, 11.0, 100.0),  # Open 缺失
        ("2024-03-05", 11.0, 13.0, 10.0, 12.0, 100.0),
        ("2024-03-06", 12.0, 14.0, 11.0, 13.0, 100.0),
        ("2024-03-07", 13.0, 15.0, 12.0, 14.0, 100.0),
        ("2024-03-08", 14.0, 16.0, 13.0, float("nan"), 100.0),  # Close 缺失
    ]
    weekly, _complete, _ = kline.aggregate_weekly(_daily(week), as_of=pd.Timestamp("2024-03-08"))
    row = weekly.iloc[0]
    assert row["Open"] == pytest.approx(11.0)  # 第一根**有效**开盘价
    assert row["Close"] == pytest.approx(14.0)  # 最后一根**有效**收盘价
    assert row["High"] == pytest.approx(16.0)
    assert row["Low"] == pytest.approx(9.0)


def test_is_week_complete_rules():
    assert kline.is_week_complete(pd.Timestamp("2024-01-05")) is True  # 周五
    assert kline.is_week_complete(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-04")) is False
    assert kline.is_week_complete(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-08")) is True


def test_aggregate_weekly_empty():
    weekly, complete, warnings = kline.aggregate_weekly(pd.DataFrame(columns=kline.STANDARD_COLUMNS))
    assert weekly.empty and complete is False and warnings


def test_weekly_summary_from_synthetic(make_ohlcv):
    daily = make_ohlcv(bars=500)
    weekly, complete, _ = kline.aggregate_weekly(daily, as_of=daily["Date"].iloc[-1] + pd.Timedelta(days=3))
    summary = indicators.weekly_summary(weekly)
    assert summary["available"] is True
    assert summary["weeks"] > 60
    assert summary["trend"] in {"上升", "下降", "震荡"}
    assert summary["rsi14"] is not None
    assert summary["ma_slow"]["ma30"] is not None


def test_weekly_summary_insufficient_weeks(make_ohlcv):
    daily = make_ohlcv(bars=60)
    weekly, _c, _w = kline.aggregate_weekly(daily, as_of=daily["Date"].iloc[-1] + pd.Timedelta(days=3))
    summary = indicators.weekly_summary(weekly)
    assert summary["ma_slow"]["ma30"] is None
    assert any("MA30" in w for w in summary["warnings"])


def test_to_csv_frame_format(sample_daily):
    out = kline.to_csv_frame(sample_daily)
    assert list(out.columns) == kline.STANDARD_COLUMNS
    assert isinstance(out["Date"].iloc[0], str)
    assert out["Date"].iloc[0][:4] == "2023"
    assert out["Close"].iloc[0] == round(sample_daily["Close"].iloc[0], 2)
