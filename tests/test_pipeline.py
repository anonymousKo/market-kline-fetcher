"""端到端流程测试（行情与 AI 全部 mock，不联网、不消耗 API 额度）。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_analysis import kline, pipeline
from stock_analysis.ai_client import LLMClient
from stock_analysis.config import AISettings, Settings
from tests.test_ai_schema import valid_payload


def _raw_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把标准 OHLCV 伪装成 yfinance 的返回（MultiIndex 列 + DatetimeIndex）。"""
    out = df.set_index("Date").copy()
    out.columns = pd.MultiIndex.from_product([list(out.columns), [symbol]])
    return out


@pytest.fixture
def patched_fetch(monkeypatch, sample_daily):
    """把网络取数替换为合成数据。"""
    frames = {"TEST": sample_daily, "SPY": sample_daily, "QQQ": sample_daily}

    def fake_fetch(symbol, period="2y", interval="1d", auto_adjust=True, start=None, end=None):
        frame = frames.get(symbol)
        if frame is None:
            return pd.DataFrame()
        return _raw_frame(frame, symbol)

    monkeypatch.setattr(kline, "fetch_ohlcv", fake_fetch)
    return frames


def _settings(tmp_path, api_key="sk-test-1234567890", benchmarks=("SPY", "QQQ")) -> Settings:
    settings = Settings()
    settings.reports_dir = str(tmp_path)
    settings.ai = AISettings(api_key=api_key, model="test-model", max_retries=0)
    settings.benchmarks = list(benchmarks)
    return settings


def _fake_llm(payload=None, fail_with=None, settings=None) -> LLMClient:
    text = json.dumps(payload if payload is not None else valid_payload(), ensure_ascii=False)

    def completion(**kwargs):
        if fail_with is not None:
            raise fail_with
        return text

    return LLMClient(settings or AISettings(api_key="sk-test-1234567890", model="test-model"), completion_fn=completion)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_run_analysis_happy_path(patched_fetch, tmp_path):
    settings = _settings(tmp_path)
    result = pipeline.run_analysis("TEST", settings=settings, client=_fake_llm(settings=settings.ai))

    assert result.ai_ok is True
    assert result.ai_error is None
    record = result.record

    # 关键字段
    assert record["symbol"] == "TEST"
    assert record["close_price"] is not None
    assert record["prompt_version"]
    assert record["model"] == "test-model"
    assert record["forward_returns"] is None  # 留给后续复核
    assert record["ai_meta"]["attempts"] == 1

    # 程序计算的指标
    assert record["indicators"]["ma"]["ma200"] is not None
    assert record["indicators"]["returns_pct"]["6m"]["value"] is not None
    assert record["weekly_summary"]["available"] is True
    assert record["quality"]["bars"] == len(patched_fetch["TEST"])

    # 相对强弱（注入了 SPY/QQQ）
    assert record["relative_strength"]["available"] is True
    assert {b["symbol"] for b in record["relative_strength"]["benchmarks"]} == {"SPY", "QQQ"}

    # AI 结构化结果
    assert record["ai_result"]["trend"]["aligned"] is True

    # 报告
    assert "# TEST 技术分析报告" in result.markdown
    for section in ("一、当前趋势", "二、动量分析", "三、成交量分析", "四、相对强弱", "五、支撑位与阻力位", "六、三种市场情景", "七、最终摘要"):
        assert section in result.markdown

    # 落盘
    assert result.paths["json"].endswith(".json")
    assert result.paths["markdown"].endswith(".md")


def test_weekly_incomplete_week_is_excluded_in_record(monkeypatch, make_ohlcv, tmp_path):
    daily = make_ohlcv(bars=80, start="2024-01-01")
    wednesdays = daily.index[daily["Date"].dt.weekday == 2]
    subset = daily.loc[: wednesdays[-1]].reset_index(drop=True)
    assert subset["Date"].iloc[-1].weekday() == 2

    monkeypatch.setattr(kline, "fetch_ohlcv", lambda symbol, **kwargs: _raw_frame(subset, symbol))

    settings = _settings(tmp_path, benchmarks=())
    result = pipeline.run_analysis(
        "TEST",
        settings=settings,
        as_of=pd.Timestamp(subset["Date"].iloc[-1]),
        use_ai=False,
    )

    record = result.record
    assert record["meta"]["weekly_last_week_complete"] is False
    assert record["weekly_summary"]["as_of"] < record["quality"]["last_bar_date"]
    assert any("尚未结束" in note for note in record["meta"]["weekly_warnings"])


# ---------------------------------------------------------------------------
# AI 不可用 / 失败时的降级
# ---------------------------------------------------------------------------


def test_missing_api_key_still_produces_report(patched_fetch, tmp_path):
    settings = _settings(tmp_path, api_key=None)
    result = pipeline.run_analysis("TEST", settings=settings)

    assert result.ai_ok is False
    assert "AI_API_KEY" in (result.ai_error or "")
    assert result.record["ai_result"] is None
    assert "# TEST 技术分析报告" in result.markdown
    assert "AI 分析未生成" in result.markdown
    assert result.record["indicators"]["ma"]["ma200"] is not None  # 指标照常计算
    assert result.paths  # 报告照常保存


def test_invalid_ai_json_is_reported_and_downgraded(patched_fetch, tmp_path):
    settings = _settings(tmp_path)

    def completion(**kwargs):
        return "抱歉，我无法输出 JSON。"

    client = LLMClient(settings.ai, completion_fn=completion)
    result = pipeline.run_analysis("TEST", settings=settings, client=client)

    assert result.ai_ok is False
    assert "JSON" in (result.ai_error or "")
    assert result.record["ai_result"] is None
    assert "AI 分析未生成" in result.markdown
    assert result.record["indicators"]["ma"]["ma50"] is not None


def test_no_ai_flag(patched_fetch, tmp_path):
    settings = _settings(tmp_path)
    result = pipeline.run_analysis("TEST", settings=settings, use_ai=False)
    assert result.ai_ok is False
    assert "--no-ai" in (result.ai_error or "")


def test_ai_raw_response_is_saved_when_parsing_fails(patched_fetch, tmp_path):
    """解析失败时必须保留模型原始返回，并说明实际返回的字段。"""
    settings = _settings(tmp_path)
    bad_text = '{"趋势": "上升", "动量": "增强"}'

    client = LLMClient(settings.ai, completion_fn=lambda **kwargs: bad_text)
    result = pipeline.run_analysis("TEST", settings=settings, client=client)

    assert result.ai_ok is False
    assert result.record["ai_raw_response"] == bad_text
    assert "trend" in (result.ai_error or "")
    assert "趋势" in (result.ai_error or "")
    assert "AI 分析未生成" in result.markdown
    # 失败也要落盘，便于排查
    assert result.paths["json"].endswith(".json")


def test_ai_meta_is_saved_when_output_is_truncated(patched_fetch, tmp_path):
    """输出被截断时，finish_reason 等元信息也要落盘，便于定位。"""
    from tests.test_ai_client import FakeResponse

    settings = _settings(tmp_path)
    truncated = json.dumps(valid_payload(), ensure_ascii=False)[:120]
    client = LLMClient(
        settings.ai, completion_fn=lambda **_: FakeResponse(truncated, finish_reason="length")
    )
    result = pipeline.run_analysis("TEST", settings=settings, client=client)

    assert result.ai_ok is False
    assert result.record["ai_meta"].get("finish_reason") == "length"
    assert "max_tokens" in (result.ai_error or "")
    assert result.record["ai_raw_response"] == truncated
    assert "AI 分析未生成" in result.markdown


def test_envelope_wrapped_result_is_recovered(patched_fetch, tmp_path):
    """模型把结果包了一层时自动解包，而不是直接报错。"""
    settings = _settings(tmp_path)
    wrapped = {"analysis": valid_payload()}

    client = LLMClient(settings.ai, completion_fn=lambda **kwargs: json.dumps(wrapped, ensure_ascii=False))
    result = pipeline.run_analysis("TEST", settings=settings, client=client)

    assert result.ai_ok is True
    assert result.record["ai_result"]["trend"]["aligned"] is True
    assert any("解包" in w for w in result.record["ai_warnings"])


# ---------------------------------------------------------------------------
# 数据异常
# ---------------------------------------------------------------------------


def test_empty_data_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(kline, "fetch_ohlcv", lambda symbol, **kwargs: pd.DataFrame())
    settings = _settings(tmp_path, benchmarks=())
    with pytest.raises(pipeline.AnalysisError) as excinfo:
        pipeline.run_analysis("NOPE", settings=settings, use_ai=False)
    assert "未获取到有效行情" in str(excinfo.value)


def test_insufficient_history_raises(monkeypatch, make_ohlcv, tmp_path):
    tiny = make_ohlcv(bars=10)
    monkeypatch.setattr(kline, "fetch_ohlcv", lambda symbol, **kwargs: _raw_frame(tiny, symbol))
    settings = _settings(tmp_path, benchmarks=())
    with pytest.raises(pipeline.AnalysisError) as excinfo:
        pipeline.run_analysis("TEST", settings=settings, use_ai=False)
    assert "历史数据不足" in str(excinfo.value)


def test_benchmark_failure_does_not_break_analysis(monkeypatch, sample_daily, tmp_path):
    def fake_fetch(symbol, **kwargs):
        if symbol == "TEST":
            return _raw_frame(sample_daily, symbol)
        raise RuntimeError("benchmark down")

    monkeypatch.setattr(kline, "fetch_ohlcv", fake_fetch)
    settings = _settings(tmp_path, benchmarks=("SPY", "QQQ"))
    result = pipeline.run_analysis("TEST", settings=settings, use_ai=False)

    relative = result.record["relative_strength"]
    assert relative["available"] is False
    assert relative["warnings"]
    assert result.record["indicators"]["ma"]["ma50"] is not None


def test_local_csv_input(monkeypatch, sample_daily, tmp_path):
    monkeypatch.setattr(kline, "fetch_ohlcv", lambda symbol, **kwargs: (_ for _ in ()).throw(AssertionError("不应联网")))
    csv_path = tmp_path / "TEST.csv"
    sample_daily.to_csv(csv_path, index=False)

    settings = _settings(tmp_path, benchmarks=())
    result = pipeline.run_analysis("TEST", settings=settings, csv_path=str(csv_path), use_ai=False)

    assert result.record["quality"]["bars"] == len(sample_daily)
    assert result.record["indicators"]["ma"]["ma200"] is not None
