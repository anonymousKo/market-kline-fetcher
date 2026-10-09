"""命令行入口。

用法（在项目根目录执行）::

    python main.py fetch   --symbol MSFT            # 仅获取行情（兼容原有 CSV 输出）
    python main.py analyze --symbol MSFT            # 获取行情 + 指标 + AI 分析
    python main.py run     --symbol MSFT            # 保存行情 + 生成分析报告
    python main.py review                           # 对历史分析记录做未来收益复核

原有的 ``python fetch_data.py --tickers ... --period ...`` 接口保持不变。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import pandas as pd

from . import pipeline, storage
from .ai_schema import preview_text
from .config import Settings, load_env

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DATA_ERROR = 2
EXIT_AI_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="股票行情抓取与 AI 技术分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py fetch --symbol MSFT\n"
            "  python main.py analyze --symbol MSFT\n"
            "  python main.py run --symbol MSFT --period 2y\n"
            "  python main.py review --min-age-days 30\n"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- fetch ----
    fetch = subparsers.add_parser("fetch", help="获取行情并保存为 CSV（兼容原有行为）")
    _add_symbol_args(fetch, multiple=True)
    fetch.add_argument("--period", default="", help="时间跨度（1mo/3mo/6mo/1y/2y...），默认 3mo")
    fetch.add_argument("--output-dir", default="", help="CSV 输出目录，默认 kline_data")
    fetch.add_argument("--limit", type=int, default=60, help="每个标的保留的 K 线数量（默认 60，与原行为一致）")

    # ---- analyze ----
    analyze = subparsers.add_parser("analyze", help="获取行情并生成技术分析报告")
    _add_symbol_args(analyze, multiple=False)
    _add_analysis_args(analyze)

    # ---- run ----
    run = subparsers.add_parser("run", help="先保存行情 CSV，再生成技术分析报告")
    _add_symbol_args(run, multiple=False)
    run.add_argument("--fetch-period", default="3mo", help="保存行情 CSV 时使用的时间跨度（默认 3mo）")
    run.add_argument("--output-dir", default="", help="CSV 输出目录，默认 kline_data")
    _add_analysis_args(run)

    # ---- review ----
    review = subparsers.add_parser("review", help="对历史分析记录计算未来收益（复核）")
    review.add_argument("--reports-dir", default="", help="报告目录，默认 reports")
    review.add_argument("--min-age-days", type=int, default=30, help="分析日至少过去多少天才复核（默认 30）")
    review.add_argument("--max-horizon", type=int, default=20, help="最长观察交易日数（默认 20）")
    review.add_argument("--force", action="store_true", help="忽略已完成标记，强制重新复核")

    return parser


def _add_symbol_args(parser: argparse.ArgumentParser, multiple: bool) -> None:
    if multiple:
        parser.add_argument("--symbol", default="", help="单个标的代码，如 MSFT")
        parser.add_argument("--tickers", default="", help="多个标的代码，英文逗号分隔（兼容原参数）")
    else:
        parser.add_argument("--symbol", required=True, help="标的代码，如 MSFT / 000001.SZ / GC=F")


def _add_analysis_args(parser: argparse.ArgumentParser, include_period: bool = True) -> None:
    if include_period:
        parser.add_argument("--period", default="", help="数据时间跨度，默认使用 ANALYSIS_PERIOD（2y）")
    parser.add_argument("--as-of", default="", help="分析日期（YYYY-MM-DD），默认使用最新数据")

    parser.add_argument("--csv", dest="csv_path", default="", help="改用本地 CSV 行情文件，不联网取数")
    parser.add_argument("--no-ai", action="store_true", help="跳过 AI 分析，只计算行情与技术指标")
    parser.add_argument("--no-benchmark", action="store_true", help="不获取 SPY/QQQ 等基准行情")
    parser.add_argument("--benchmarks", default="", help="自定义基准，英文逗号分隔，如 SPY,QQQ")
    parser.add_argument("--reports-dir", default="", help="报告输出目录，默认 reports")
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不写入文件")
    parser.add_argument("--print-report", action="store_true", help="在终端打印完整 Markdown 报告")
    parser.add_argument("--print-json", action="store_true", help="在终端打印结构化 JSON 结果")


# ---------------------------------------------------------------------------
# 各子命令实现
# ---------------------------------------------------------------------------


def _resolve_symbols(args: argparse.Namespace, default: str = "") -> List[str]:
    raw_parts: List[str] = []
    for value in (getattr(args, "symbol", ""), getattr(args, "tickers", "")):
        if value:
            raw_parts.extend(part for part in str(value).split(",") if part.strip())
    if not raw_parts and default:
        raw_parts = [part for part in default.split(",") if part.strip()]
    return [part.strip().upper() for part in raw_parts if part.strip()]


def _cmd_fetch(args: argparse.Namespace) -> int:
    import fetch_data  # 复用原有抓取脚本，保持 CSV 格式与 GITHUB_STEP_SUMMARY 行为

    symbols = _resolve_symbols(args, default="GC=F,MU,NVDA,SOXQ")
    if not symbols:
        logger.error("未指定标的，请使用 --symbol 或 --tickers")
        return EXIT_USAGE

    period = args.period.strip() or "3mo"
    output_dir = args.output_dir.strip() or "kline_data"
    written = fetch_data.run_fetch(symbols, period, output_dir=output_dir, limit=args.limit)
    logger.info("共写出 %d 个 CSV 文件到 %s/", len(written), output_dir)
    return EXIT_OK


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "reports_dir", ""):
        settings.reports_dir = args.reports_dir.strip()
    if getattr(args, "benchmarks", ""):
        settings.benchmarks = [item.strip().upper() for item in args.benchmarks.split(",") if item.strip()]
    if getattr(args, "no_benchmark", False):
        settings.benchmarks = []
    return settings


def _run_analysis(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    symbol = args.symbol.strip().upper()

    as_of: Optional[pd.Timestamp] = None
    if getattr(args, "as_of", ""):
        try:
            as_of = pd.Timestamp(args.as_of.strip())
        except (ValueError, TypeError):
            logger.error("--as-of 参数格式非法：%s（应为 YYYY-MM-DD）", args.as_of)
            return EXIT_USAGE

    try:
        result = pipeline.run_analysis(
            symbol,
            settings=settings,
            period=getattr(args, "period", "") or None,
            as_of=as_of,
            csv_path=getattr(args, "csv_path", "") or None,
            use_ai=not getattr(args, "no_ai", False),
            save=not getattr(args, "no_save", False),
        )
    except pipeline.AnalysisError as exc:
        logger.error("%s", exc)
        return EXIT_DATA_ERROR

    record = result.record
    logger.info(
        "分析完成：%s 收盘 %s（数据截止 %s）",
        record.get("symbol"),
        record.get("close_price"),
        (record.get("quality") or {}).get("last_bar_date"),
    )
    for issue in (record.get("quality") or {}).get("issues", []):
        logger.warning("数据问题：%s", issue)

    if result.paths:
        logger.info("Markdown 报告：%s", result.paths.get("markdown"))
        logger.info("结构化结果：%s", result.paths.get("json"))

    if getattr(args, "print_report", False):
        print("\n" + result.markdown)
    if getattr(args, "print_json", False):
        payload = {k: v for k, v in record.items() if k != "markdown"}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    if not result.ai_ok:
        logger.error("AI 分析未生成：%s", result.ai_error)
        raw = record.get("ai_raw_response")
        if raw:
            if result.paths.get("json"):
                logger.error(
                    "模型原始返回内容已保存，请查看该文件的 ai_raw_response 字段：%s",
                    result.paths["json"],
                )
            else:
                logger.error("模型原始返回（截断显示）：%s", preview_text(raw, 500))
        # 显式使用 --no-ai 时不算错误
        return EXIT_OK if getattr(args, "no_ai", False) else EXIT_AI_ERROR
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    return _run_analysis(args)


def _cmd_run(args: argparse.Namespace) -> int:
    import fetch_data

    symbol = args.symbol.strip().upper()
    output_dir = args.output_dir.strip() or "kline_data"
    fetch_period = args.fetch_period.strip() or "3mo"

    # 先保存行情 CSV（失败不阻塞分析）
    try:
        written = fetch_data.run_fetch([symbol], fetch_period, output_dir=output_dir, limit=60)
        if written:
            logger.info("行情已保存：%s", written[0])
    except Exception as exc:
        logger.warning("行情 CSV 保存失败（不影响后续分析）：%s", exc)

    return _run_analysis(args)


def _cmd_review(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    reports_dir = args.reports_dir.strip() or settings.reports_dir

    logger.info(
        "开始复核 %s 中的历史分析记录（min_age_days=%d, max_horizon=%d）...",
        reports_dir,
        args.min_age_days,
        args.max_horizon,
    )
    updates = storage.review_records(
        reports_dir,
        min_age_days=args.min_age_days,
        max_horizon=args.max_horizon,
        force=args.force,
    )
    if not updates:
        logger.info("没有需要复核的记录（可能都已完成，或分析日距今不足 %d 天）", args.min_age_days)
        return EXIT_OK

    for update in updates:
        forward = update["forward_returns"]
        horizons = forward.get("horizons") or {}
        if horizons:
            summary = "，".join(
                "{}: {}".format(
                    key,
                    "N/A" if (value or {}).get("return") is None else f"{(value or {}).get('return'):+.2%}",
                )
                for key, value in horizons.items()
            )
        else:
            summary = str(forward.get("error") or "暂无可用数据")
        logger.info(
            "%s：%s；MFE=%s，MAE=%s",
            update["symbol"],
            summary,
            _fmt_pct(forward.get("max_favorable_excursion")),
            _fmt_pct(forward.get("max_adverse_excursion")),
        )
    logger.info("本次复核完成 %d 条记录。", len(updates))
    return EXIT_OK


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:+.2%}"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    pipeline.setup_logging(args.log_level)

    handlers = {
        "fetch": _cmd_fetch,
        "analyze": _cmd_analyze,
        "run": _cmd_run,
        "review": _cmd_review,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("操作被用户中断")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
