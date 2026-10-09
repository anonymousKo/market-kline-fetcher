"""端到端分析流程编排。

职责边界（严格分离）：

1. :mod:`stock_analysis.kline` —— 行情获取与清洗（不含指标、不含 AI）
2. :mod:`stock_analysis.indicators` / :mod:`stock_analysis.key_levels` —— 纯计算
3. :mod:`stock_analysis.context` —— 组织给模型的数据
4. :mod:`stock_analysis.ai_client` —— 模型 API 调用
5. :mod:`stock_analysis.report` / :mod:`stock_analysis.storage` —— 输出与持久化

本模块只负责按顺序调用它们，并在**任一环节失败时给出明确错误**。
特别注意：AI 调用失败**不会**影响行情获取与指标计算，报告仍会生成（缺少 AI 部分）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import ai_schema, context as context_builder, indicators, key_levels, kline, report, storage
from .ai_client import AIConfigError, AIError, LLMClient, dumps_payload
from .config import AISettings, Settings
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_INSTRUCTION

logger = logging.getLogger(__name__)

# 少于该数量时无法进行有意义的技术分析，直接报错
MIN_REQUIRED_BARS = 20

# 落盘保存的模型原始返回内容上限（字符），便于出错时排查
MAX_RAW_RESPONSE_CHARS = 20000


def _truncate_raw(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    if len(text) <= MAX_RAW_RESPONSE_CHARS:
        return text
    return text[:MAX_RAW_RESPONSE_CHARS] + f"\n...(已截断，模型原始返回共 {len(text)} 字符)"


class AnalysisError(RuntimeError):
    """分析流程中的可预期错误（行情获取失败、数据不足等）。"""


@dataclass
class AnalysisResult:
    """一次分析的结果。"""

    record: Dict[str, Any]
    markdown: str
    paths: Dict[str, str] = field(default_factory=dict)
    ai_ok: bool = False
    ai_error: Optional[str] = None


def setup_logging(level: str = "INFO") -> None:
    """统一日志格式（不输出任何密钥）。"""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))


# ---------------------------------------------------------------------------
# 行情准备
# ---------------------------------------------------------------------------


def _period_start(period: str, as_of: pd.Timestamp) -> str:
    """把 yfinance 的 period 语法换算为起始日期（用于按历史日期分析）。"""
    mapping_days = {
        "1mo": 31,
        "3mo": 92,
        "6mo": 183,
        "1y": 366,
        "2y": 732,
        "5y": 1830,
        "10y": 3660,
    }
    days = mapping_days.get(str(period).strip().lower(), 732)
    return (as_of - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def load_daily_data(
    symbol: str,
    settings: Settings,
    period: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
    csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, kline.DataQuality]:
    """获取并检查日线数据（不使用 AI）。"""
    period = period or settings.analysis_period

    if csv_path:
        logger.info("从本地文件读取行情：%s", csv_path)
        raw = kline.read_csv_file(csv_path)
    else:
        start = end = None
        if as_of is not None:
            start = _period_start(period, as_of)
            end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("按指定分析日期取数：%s ~ %s", start, end)
        logger.info("抓取 %s 日线行情（period=%s）...", symbol, period)
        try:
            raw = kline.fetch_ohlcv(symbol, period=period, interval="1d", start=start, end=end)
        except Exception as exc:  # 网络/数据源异常 -> 明确报错
            raise AnalysisError(f"获取 {symbol} 行情失败：{type(exc).__name__}: {exc}") from exc

    daily, quality = kline.prepare_daily(
        raw,
        symbol,
        min_bars=settings.min_daily_bars,
        timezone_name=settings.timezone,
        as_of=as_of,
    )

    if daily is None or daily.empty:
        raise AnalysisError(
            f"{symbol} 未获取到有效行情数据。请检查代码是否正确（yfinance 格式，如 MSFT / 000001.SZ / GC=F），"
            "或稍后重试。"
        )
    if len(daily) < MIN_REQUIRED_BARS:
        raise AnalysisError(
            f"{symbol} 历史数据不足：仅 {len(daily)} 根日 K，至少需要 {MIN_REQUIRED_BARS} 根才能进行技术分析。"
        )
    return daily, quality


def build_benchmarks(
    benchmark_symbols: List[str],
    settings: Settings,
    period: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """抓取基准行情；单个基准失败不影响整体流程。"""
    frames: Dict[str, pd.DataFrame] = {}
    for bench in benchmark_symbols:
        try:
            raw = kline.fetch_ohlcv(bench, period=period or settings.analysis_period, interval="1d")
            bench_df, bench_quality = kline.prepare_daily(
                raw,
                bench,
                min_bars=MIN_REQUIRED_BARS,
                timezone_name=settings.timezone,
                as_of=as_of,
            )
            if bench_df.empty:
                logger.warning("基准 %s 未获取到数据，已跳过", bench)
                continue
            for issue in bench_quality.issues:
                logger.warning("基准 %s 数据问题：%s", bench, issue)
            frames[bench] = bench_df
        except Exception as exc:
            logger.warning("获取基准 %s 行情失败，已跳过相对强弱计算：%s", bench, exc)
    return frames


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_analysis(
    symbol: str,
    settings: Optional[Settings] = None,
    period: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
    csv_path: Optional[str] = None,
    use_ai: bool = True,
    benchmarks: Optional[List[str]] = None,
    save: bool = True,
    client: Optional[LLMClient] = None,
) -> AnalysisResult:
    """对单个标的执行完整分析流程。

    AI 失败不会抛出异常：会在结果中记录 ``ai_error`` 并生成不含 AI 部分的报告。
    行情获取失败或数据严重不足时抛出 :class:`AnalysisError`。
    """
    settings = settings or Settings.from_env()
    symbol = symbol.strip().upper()
    if not symbol:
        raise AnalysisError("股票代码不能为空")

    analysis_date = (
        pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
        if as_of is not None
        else None
    )

    daily, quality = load_daily_data(
        symbol, settings, period=period, as_of=as_of, csv_path=csv_path
    )
    if analysis_date is None:
        analysis_date = quality.last_bar_date or datetime.now().strftime("%Y-%m-%d")

    # ---- 周线聚合（自动剔除进行中的交易周） ----
    weekly, last_week_complete, weekly_warnings = kline.aggregate_weekly(
        daily, as_of=as_of, drop_incomplete=True, timezone_name=settings.timezone
    )
    for warning in weekly_warnings:
        logger.info("周线聚合：%s", warning)

    # ---- 指标与价格结构 ----
    daily_snapshot = indicators.daily_indicator_snapshot(daily)
    weekly_snapshot = indicators.weekly_summary(weekly)
    levels = key_levels.build_key_levels(daily)

    # ---- 相对强弱 ----
    benchmark_symbols = benchmarks if benchmarks is not None else list(settings.benchmarks)
    benchmark_frames = build_benchmarks(benchmark_symbols, settings, period=period, as_of=as_of)
    relative = context_builder.build_relative_strength(daily, benchmark_frames)

    # ---- 组装给模型的上下文 ----
    market, currency = kline.infer_market(symbol)
    model_context = context_builder.build_context(
        symbol=symbol,
        market=market,
        currency=currency,
        analysis_date=analysis_date,
        daily_df=daily,
        daily_snapshot=daily_snapshot,
        weekly_df=weekly,
        weekly_snapshot=weekly_snapshot,
        key_levels=levels,
        quality=quality.to_dict(),
        relative_strength=relative,
        prompt_version=PROMPT_VERSION,
        adjustment_note=kline.ADJUSTMENT_NOTE,
        timezone_name=settings.timezone,
    )

    # ---- AI 分析（失败不影响主流程） ----
    ai_result: Optional[Dict[str, Any]] = None
    ai_warnings: List[str] = []
    ai_meta: Dict[str, Any] = {}
    ai_error: Optional[str] = None
    ai_raw: Optional[str] = None
    model_name: Optional[str] = None

    if not use_ai:
        ai_error = "已通过 --no-ai 参数跳过 AI 分析"
        logger.info("按参数要求跳过 AI 分析")
    elif not settings.ai.configured:
        ai_error = (
            "未配置大模型 API 密钥，已跳过 AI 分析（行情获取与技术指标计算正常完成）。"
            "请设置环境变量 AI_API_KEY，参考 .env.example。"
        )
        logger.warning(ai_error)
    else:
        model_name = settings.ai.model
        llm = client or LLMClient(settings.ai)
        try:
            logger.info("调用大模型进行技术面分析：model=%s", settings.ai.model)
            ai_result, ai_warnings, ai_meta = llm.complete_json(
                SYSTEM_PROMPT, USER_INSTRUCTION + dumps_payload(model_context)
            )
            ai_raw = ai_meta.get("raw_response")
            logger.info(
                "AI 分析完成（耗时 %.2fs，tokens=%s）",
                ai_meta.get("elapsed_seconds", 0),
                ai_meta.get("total_tokens", "未知"),
            )
        except AIConfigError as exc:
            ai_error = str(exc)
            logger.warning("%s", ai_error)
        except ai_schema.AIResultError as exc:
            ai_raw = getattr(exc, "raw_response", None)
            ai_meta = dict(getattr(exc, "meta", None) or {})
            ai_error = f"模型返回内容不是合法的结构化 JSON：{exc}"
            logger.error("%s", ai_error)
        except AIError as exc:
            ai_error = str(exc)
            logger.error("AI 调用失败：%s", ai_error)
        except Exception as exc:  # 兜底，保证主流程不被 AI 拖垮
            ai_error = f"AI 分析出现未预期错误：{type(exc).__name__}: {exc}"
            logger.exception("AI 分析出现未预期错误")

    # ---- 组装记录 ----
    record: Dict[str, Any] = {
        "record_version": storage.RECORD_VERSION,
        "symbol": symbol,
        "market": market,
        "currency": currency,
        "analysis_date": analysis_date,
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "close_price": daily_snapshot.get("close"),
        "model": settings.ai.model if model_name else None,
        "prompt_version": PROMPT_VERSION,
        "quality": quality.to_dict(),
        "indicators": daily_snapshot,
        "weekly_summary": weekly_snapshot,
        "key_levels": levels,
        "relative_strength": relative,
        "ai_result": ai_result,
        "ai_warnings": ai_warnings,
        "ai_meta": {k: v for k, v in ai_meta.items() if k != "raw_response"},
        "ai_raw_response": _truncate_raw(ai_raw),
        "ai_error": ai_error,
        "context": model_context,
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timezone": settings.timezone,
            "weekly_last_week_complete": last_week_complete,
            "weekly_warnings": weekly_warnings,
            "benchmarks": benchmark_symbols,
        },
        "forward_returns": None,  # 由 storage.review_records 后续填充
    }

    markdown = report.render_markdown(record)
    record["markdown"] = markdown

    result = AnalysisResult(
        record=record,
        markdown=markdown,
        ai_ok=ai_result is not None,
        ai_error=ai_error,
    )

    if save:
        result.paths = storage.save_analysis(record, settings.reports_dir)

    return result
