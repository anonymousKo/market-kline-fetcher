"""原有行情获取功能未被破坏的回归测试（通过 mock 数据源，不联网）。"""

from __future__ import annotations

import os

import pandas as pd
import pytest
import yfinance

import fetch_data
from stock_analysis import kline


def _fake_download(sample_daily):
    def fake(symbol, **kwargs):
        df = sample_daily.set_index("Date").copy()
        # 模拟 yfinance 单标的返回的 MultiIndex 列
        df.columns = pd.MultiIndex.from_product([list(df.columns), [symbol]])
        return df

    return fake


@pytest.fixture
def patched_download(monkeypatch, sample_daily):
    monkeypatch.setattr(yfinance, "download", _fake_download(sample_daily))


def test_fetch_single_ticker_output_format(patched_download, sample_daily):
    df = fetch_data.fetch_single_ticker("MSFT", period="2y")

    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 60  # 与原行为一致：只保留最后 60 根
    assert isinstance(df["Date"].iloc[0], str)
    assert df["Date"].iloc[0] == sample_daily["Date"].iloc[-60].strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        assert df[col].iloc[0] == round(sample_daily[col].iloc[-60], 2)


def test_fetch_single_ticker_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(yfinance, "download", lambda symbol, **kwargs: pd.DataFrame())
    assert fetch_data.fetch_single_ticker("NOPE").empty


def test_fetch_single_ticker_error_is_tolerated(monkeypatch):
    def boom(symbol, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(yfinance, "download", boom)
    assert fetch_data.fetch_single_ticker("NOPE").empty


def test_run_fetch_writes_csv_and_summary(patched_download, tmp_path):
    output_dir = tmp_path / "kline_data"
    summary = tmp_path / "summary.md"

    written = fetch_data.run_fetch(
        ["MSFT", "GC=F"],
        period="3mo",
        output_dir=str(output_dir),
        limit=10,
        summary_path=str(summary),
    )

    assert len(written) == 2
    assert os.path.basename(written[0]) == "MSFT_kline.csv"
    assert os.path.basename(written[1]) == "GC_F_kline.csv"  # 与原实现一致：'=' 会被替换为 '_'

    content = (output_dir / "MSFT_kline.csv").read_text(encoding="utf-8")
    assert content.splitlines()[0] == "Date,Open,High,Low,Close,Volume"
    assert len(content.strip().splitlines()) == 11  # 表头 + 10 行

    summary_text = summary.read_text(encoding="utf-8")
    assert "每日 K 线数据汇总" in summary_text
    assert "点击展开查看：MSFT" in summary_text
    assert "```csv" in summary_text


def test_run_fetch_respects_github_step_summary_env(monkeypatch, patched_download, tmp_path):
    summary = tmp_path / "gh_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    fetch_data.run_fetch(["MSFT"], period="3mo", output_dir=str(tmp_path / "out"), limit=5)
    assert summary.exists()
    assert "MSFT" in summary.read_text(encoding="utf-8")


def test_parse_args_defaults_preserved():
    tickers, period = fetch_data.parse_args([])
    assert tickers == ["GC=F", "MU", "NVDA", "SOXQ"]
    assert period == "3mo"

    tickers, period = fetch_data.parse_args(["--tickers", "aapl, msft", "--period", "1y"])
    assert tickers == ["AAPL", "MSFT"]
    assert period == "1y"


def test_clean_ticker_name_behaviour_unchanged():
    # 原实现的正则包含 '=' 与 '|'，行为必须保持一致
    assert fetch_data.clean_ticker_name("GC=F") == "GC_F"
    assert fetch_data.clean_ticker_name("A/B") == "A_B"
    assert fetch_data.clean_ticker_name("A:B|C") == "A_B_C"


def test_kline_read_csv_file_roundtrip(patched_download, tmp_path):
    df = fetch_data.fetch_single_ticker("MSFT", period="2y")
    path = tmp_path / "msft.csv"
    df.to_csv(path, index=False)

    raw = kline.read_csv_file(str(path))
    normalized = kline.normalize_ohlcv(raw)
    assert len(normalized) == len(df)
    assert list(normalized.columns) == kline.STANDARD_COLUMNS

    with pytest.raises(FileNotFoundError):
        kline.read_csv_file(str(tmp_path / "missing.csv"))
