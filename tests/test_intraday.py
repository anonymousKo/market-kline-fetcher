"""盘中（未收盘）数据处理测试。

覆盖：
* 交易所时区推断；
* “当日是否已收盘”的判定（含跨时区、收盘结算缓冲、周末）；
* 把盘中那根 K 线拆出计算之外；
* 端到端：默认策略下指标不再被盘中值污染，且报告/上下文明确标注。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_analysis import kline, pipeline
from stock_analysis.ai_client import LLMClient
from stock_analysis.config import AISettings, Settings

NY = "America/New_York"
MOCK_TODAY = "2026-10-09"  # 周五
MID_SESSION = pd.Timestamp("2026-10-09 12:43", tz=NY)


# ---------------------------------------------------------------------------
# 时区推断
# ---------------------------------------------------------------------------


def test_infer_timezone_by_market():
    assert kline.infer_timezone("MSFT") == NY
    assert kline.infer_timezone("SOXQ") == NY
    assert kline.infer_timezone("000001.SZ") == "Asia/Shanghai"
    assert kline.infer_timezone("600000.SS") == "Asia/Shanghai"
    assert kline.infer_timezone("0700.HK") == "Asia/Hong_Kong"
    assert kline.infer_timezone("7203.T") == "Asia/Tokyo"
    assert kline.infer_timezone("SOMETHING") == NY
    assert kline.infer_timezone("EURUSD=X") == "UTC"


# ---------------------------------------------------------------------------
# 会话是否进行中
# ---------------------------------------------------------------------------


def test_session_in_progress_boundaries():
    day = pd.Timestamp(MOCK_TODAY)

    in_progress, reason = kline.session_in_progress(
        day, NY, now=pd.Timestamp("2026-10-09 12:43", tz=NY)
    )
    assert in_progress is True and "尚未收盘" in (reason or "")

    # 刚收盘 15 分钟：数据源可能还在结算，保守视为未完成
    in_progress, reason = kline.session_in_progress(
        day, NY, now=pd.Timestamp("2026-10-09 16:15", tz=NY)
    )
    assert in_progress is True and "结算" in (reason or "")

    # 收盘 + 结算缓冲之后：已收盘
    in_progress, _ = kline.session_in_progress(day, NY, now=pd.Timestamp("2026-10-09 17:00", tz=NY))
    assert in_progress is False


def test_session_in_progress_uses_exchange_timezone_not_machine_timezone():
    """复现真实故障：本机已跨天，但交易所仍是盘中。"""
    machine_time = pd.Timestamp("2026-10-10 00:43", tz="UTC")  # 纽约当地 10-09 20:43
    in_progress, _ = kline.session_in_progress(pd.Timestamp(MOCK_TODAY), NY, now=machine_time)
    assert in_progress is False  # 纽约 20:43 其实已收盘

    # 真正的盘中：UTC 16:43 == 纽约 12:43
    in_progress, _ = kline.session_in_progress(
        pd.Timestamp(MOCK_TODAY), NY, now=pd.Timestamp("2026-10-09 16:43", tz="UTC")
    )
    assert in_progress is True


def test_session_in_progress_weekend_and_older_bar():
    assert kline.session_in_progress(pd.Timestamp("2026-10-03"), NY, now=MID_SESSION)[0] is False
    # 周六：当地今天不是交易日，最后一根周五 K 线视为已收
    saturday = pd.Timestamp("2026-10-10 12:00", tz=NY)
    assert kline.session_in_progress(pd.Timestamp(MOCK_TODAY), NY, now=saturday)[0] is False


# ---------------------------------------------------------------------------
# 拆分盘中 K 线
# ---------------------------------------------------------------------------


def _daily_ending_today(make_ohlcv):
    """构造以 MOCK_TODAY 为最后一根、最后一根为明显“盘中值”的日线。"""
    base = make_ohlcv(bars=120)
    base["Date"] = pd.bdate_range(end=MOCK_TODAY, periods=len(base))
    base.loc[base.index[-1], "Volume"] = 100  # 盘中累计量，远小于均量
    base.loc[base.index[-1], "Close"] = float(base["Close"].iloc[-2]) * 1.5  # 盘中价
    return base


def test_split_incomplete_bar(make_ohlcv):
    daily = _daily_ending_today(make_ohlcv)
    last_close = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2])

    quality = kline.check_quality(daily, "TEST", min_bars=20, as_of=MID_SESSION, timezone_name=NY)
    assert quality.source_last_bar_incomplete is True

    closed, snapshot = kline.split_incomplete_bar(daily, quality)

    assert len(closed) == len(daily) - 1
    assert pd.Timestamp(closed["Date"].iloc[-1]).strftime("%Y-%m-%d") == "2026-10-08"

    assert snapshot is not None
    assert snapshot["date"] == MOCK_TODAY
    assert snapshot["last"] == pytest.approx(round(last_close, 4))
    assert snapshot["prev_close"] == pytest.approx(prev_close)
    assert snapshot["change_pct_vs_prev_close"] == pytest.approx(
        round((last_close / prev_close - 1) * 100, 2)
    )
    assert snapshot["volume"] == pytest.approx(100.0)
    assert "未参与" in snapshot["note"]

    # 质量对象改为描述“实际用于分析的数据”
    assert quality.intraday_excluded is True
    assert quality.incomplete_bar is False
    assert quality.bars == len(daily) - 1
    assert quality.last_bar_date == "2026-10-08"
    assert quality.source_last_bar_date == MOCK_TODAY
    assert any("排除出指标计算" in w for w in quality.warnings)


def test_split_incomplete_bar_noop_cases(make_ohlcv):
    daily = _daily_ending_today(make_ohlcv)

    # 非盘中：原样返回
    closed_quality = kline.check_quality(
        daily, "TEST", min_bars=20, as_of=pd.Timestamp("2026-10-12 10:00", tz=NY), timezone_name=NY
    )
    same, snapshot = kline.split_incomplete_bar(daily, closed_quality)
    assert snapshot is None and len(same) == len(daily)

    # 只有一根 K 线：无法拆分，保留原样
    single = daily.tail(1).reset_index(drop=True)
    quality = kline.check_quality(single, "TEST", min_bars=1, as_of=MID_SESSION, timezone_name=NY)
    same, snapshot = kline.split_incomplete_bar(single, quality)
    assert snapshot is None and len(same) == 1


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def _raw_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.set_index("Date").copy()
    out.columns = pd.MultiIndex.from_product([list(out.columns), [symbol]])
    return out


@pytest.fixture
def patched_intraday_fetch(monkeypatch, make_ohlcv):
    frame = _daily_ending_today(make_ohlcv)

    def fake_fetch(symbol, period="2y", interval="1d", auto_adjust=True, start=None, end=None):
        return _raw_frame(frame, symbol)

    monkeypatch.setattr(kline, "fetch_ohlcv", fake_fetch)
    return frame


def _settings(tmp_path) -> Settings:
    settings = Settings()
    settings.reports_dir = str(tmp_path)
    settings.benchmarks = []
    settings.ai = AISettings(api_key="sk-test-1234567890", model="test-model")
    return settings


def test_pipeline_excludes_intraday_bar_by_default(patched_intraday_fetch, tmp_path):
    frame = patched_intraday_fetch
    closed_close = float(frame["Close"].iloc[-2])
    closed_volume = float(frame["Volume"].iloc[-2])

    result = pipeline.run_analysis(
        "TEST", settings=_settings(tmp_path), now=MID_SESSION, use_ai=False
    )
    record = result.record

    # 指标必须基于“最后已收盘”那根，而不是盘中值
    assert record["quality"]["intraday_excluded"] is True
    assert record["quality"]["last_bar_date"] == "2026-10-08"
    assert record["quality"]["source_last_bar_date"] == MOCK_TODAY
    assert record["close_price"] == pytest.approx(closed_close, rel=1e-6)
    assert record["indicators"]["close"] == pytest.approx(closed_close, rel=1e-6)
    assert record["indicators"]["volume"]["volume"] == pytest.approx(closed_volume, rel=1e-6)

    # 盘中快照单独保存，且不进入日 K 序列
    intraday = record["intraday"]
    assert intraday is not None and intraday["date"] == MOCK_TODAY
    sent_bars = record["context"]["price"]["recent_daily_bars"]
    assert all(bar["date"] != MOCK_TODAY for bar in sent_bars)
    assert sent_bars[-1]["date"] == "2026-10-08"

    # 上下文明确告知模型
    quality_ctx = record["context"]["data_quality"]
    assert quality_ctx["intraday_excluded"] is True
    assert "排除" in quality_ctx["intraday_note"]
    assert record["context"]["price"]["intraday"]["last"] is not None

    # 报告标注
    assert "盘中数据" in result.markdown
    assert "已排除出指标计算" in result.markdown
    assert "盘中最新价（未参与指标计算）" in result.markdown

    # analysis_date 取最后已收盘交易日，保证后续复核不会从盘中价起算
    assert record["analysis_date"] == "2026-10-08"
    assert record["meta"]["intraday_policy"] == "drop"


def test_pipeline_keep_policy_keeps_bar_with_warning(patched_intraday_fetch, tmp_path):
    frame = patched_intraday_fetch

    result = pipeline.run_analysis(
        "TEST",
        settings=_settings(tmp_path),
        now=MID_SESSION,
        use_ai=False,
        intraday_policy="keep",
    )
    record = result.record

    assert record["intraday"] is None
    assert record["quality"]["bars"] == len(frame)
    assert record["quality"]["source_last_bar_incomplete"] is True
    assert record["close_price"] == pytest.approx(float(frame["Close"].iloc[-1]), rel=1e-6)
    assert any("keep" in w for w in record["quality"]["warnings"])
    assert "keep" in json.dumps(record["quality"], ensure_ascii=False)


def test_pipeline_intraday_config_from_env(patched_intraday_fetch, tmp_path, monkeypatch):
    monkeypatch.setenv("INTRADAY_POLICY", "keep")
    settings = Settings.from_env()
    settings.reports_dir = str(tmp_path)
    settings.benchmarks = []
    settings.ai = AISettings(api_key="sk-test-1234567890", model="test-model")

    result = pipeline.run_analysis("TEST", settings=settings, now=MID_SESSION, use_ai=False)
    assert result.record["intraday"] is None
    assert result.record["meta"]["intraday_policy"] == "keep"


def test_intraday_data_reaches_the_ai_payload(patched_intraday_fetch, tmp_path):
    """AI 侧也要能看到盘中价，并知道它未参与计算。"""
    from tests.test_ai_schema import valid_payload

    settings = _settings(tmp_path)
    client = LLMClient(
        settings.ai,
        completion_fn=lambda **_: json.dumps(valid_payload(), ensure_ascii=False),
    )
    result = pipeline.run_analysis("TEST", settings=settings, now=MID_SESSION, client=client)

    assert result.ai_ok is True
    assert result.record["ai_result"]["trend"]["aligned"] is True
    assert result.record["context"]["price"]["intraday"]["date"] == MOCK_TODAY
