"""pytest 公共 fixture：生成确定性的合成行情，测试中**不联网**、**不消耗 API 额度**。"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

SYMBOL = "TEST"


def _make_ohlcv(
    bars: int = 300,
    start: str = "2023-01-02",
    seed: int = 7,
    drift: float = 0.0005,
    sigma: float = 0.012,
    price: float = 100.0,
    volume_range=(1_000_000, 5_000_000),
) -> pd.DataFrame:
    """生成确定性的日线 OHLCV（仅工作日）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=bars)
    returns = rng.normal(drift, sigma, bars)
    close = price * np.exp(np.cumsum(returns))
    prev_close = np.concatenate([[price], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, 0.002, bars))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, bars))) + 0.01
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, bars))) - 0.01
    volume = rng.integers(volume_range[0], volume_range[1], bars)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        }
    )


@pytest.fixture
def make_ohlcv():
    """返回合成行情生成函数（可自定义参数）。"""
    return _make_ohlcv


@pytest.fixture
def sample_daily() -> pd.DataFrame:
    """300 根日 K（足够计算 MA200 与 6 个月收益率）。"""
    return _make_ohlcv(bars=300)


def make_daily_from_closes(closes: List[float], start: str = "2024-01-01") -> pd.DataFrame:
    """用给定的收盘价序列构造日线（其余 OHLCV 由收盘价推导），用于精确断言。"""
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
