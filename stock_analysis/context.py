"""构建发送给大模型的结构化上下文。

原则：
* 只发送**分析所需**的摘要与近期 K 线，完整历史数据保留在本地。
* 数值全部由程序计算得到，大模型只负责解释。
* 明确标注数据质量问题、缺失项与不可用指标，避免模型"脑补"。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 发送给模型的近期 K 线数量（避免无谓地消耗 token）
RECENT_DAILY_BARS = 30
RECENT_WEEKLY_BARS = 12


def _clean(value: object, digits: int = 4) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def _date_str(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def daily_bars_payload(df: pd.DataFrame, limit: int = RECENT_DAILY_BARS) -> List[Dict[str, object]]:
    """近期日 K 线（紧凑格式）。"""
    if df is None or df.empty:
        return []
    tail = df.tail(limit)
    bars: List[Dict[str, object]] = []
    for _, row in tail.iterrows():
        bars.append(
            {
                "date": _date_str(row["Date"]),
                "open": _clean(row["Open"], 2),
                "high": _clean(row["High"], 2),
                "low": _clean(row["Low"], 2),
                "close": _clean(row["Close"], 2),
                "volume": _clean(row["Volume"], 0),
            }
        )
    return bars


def weekly_bars_payload(df: pd.DataFrame, limit: int = RECENT_WEEKLY_BARS) -> List[Dict[str, object]]:
    """近期周 K 线（紧凑格式，仅包含已结束的交易周）。"""
    if df is None or df.empty:
        return []
    tail = df.tail(limit)
    bars: List[Dict[str, object]] = []
    for _, row in tail.iterrows():
        bars.append(
            {
                "week_end": _date_str(row.get("WeekEnd", row["Date"])),
                "open": _clean(row["Open"], 2),
                "high": _clean(row["High"], 2),
                "low": _clean(row["Low"], 2),
                "close": _clean(row["Close"], 2),
                "volume": _clean(row["Volume"], 0),
                "bars": int(row["Bars"]) if "Bars" in row and pd.notna(row["Bars"]) else None,
            }
        )
    return bars


def build_context(
    symbol: str,
    market: str,
    currency: str,
    analysis_date: str,
    daily_df: pd.DataFrame,
    daily_snapshot: Dict[str, object],
    weekly_df: pd.DataFrame,
    weekly_snapshot: Dict[str, object],
    key_levels: Dict[str, object],
    quality: Dict[str, object],
    relative_strength: Optional[Dict[str, object]] = None,
    prompt_version: str = "",
    adjustment_note: str = "",
    timezone_name: str = "",
) -> Dict[str, object]:
    """组装发送给大模型的 JSON（不包含任何密钥）。"""
    warnings: List[str] = []
    warnings.extend(str(item) for item in (daily_snapshot.get("warnings") or []))
    warnings.extend(str(item) for item in (weekly_snapshot.get("warnings") or []))

    data_cutoff = quality.get("last_bar_date")
    weekly_bars = weekly_bars_payload(weekly_df)

    context: Dict[str, object] = {
        "meta": {
            "symbol": symbol,
            "market": market,
            "currency": currency,
            "analysis_date": analysis_date,
            "data_cutoff_date": data_cutoff,
            "timezone": timezone_name,
            "adjustment": adjustment_note,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt_version": prompt_version,
        },
        "data_quality": {
            "daily_bars": quality.get("bars"),
            "weekly_bars": int(len(weekly_df)) if weekly_df is not None and not weekly_df.empty else 0,
            "start_date": quality.get("start_date"),
            "end_date": quality.get("end_date"),
            "issues": list(quality.get("issues") or []),
            "warnings": list(quality.get("warnings") or []),
            "indicator_warnings": warnings,
            "last_bar_incomplete": bool(quality.get("incomplete_bar")),
            "adjusted": quality.get("adjusted"),
            "adjustment_note": quality.get("adjustment_note"),
            "weekly_notes": list(weekly_df.attrs.get("warnings", [])) if hasattr(weekly_df, "attrs") else [],
            "weekly_last_week_complete": bool(weekly_df.attrs.get("last_week_complete", True))
            if hasattr(weekly_df, "attrs")
            else True,
        },
        "price": {
            "close": daily_snapshot.get("close"),
            "recent_daily_bars": daily_bars_payload(daily_df),
            "recent_weekly_bars": weekly_bars,
            "recent_daily_bars_note": f"最近 {len(daily_bars_payload(daily_df))} 个交易日的日 K 线",
            "recent_weekly_bars_note": (
                f"最近 {len(weekly_bars)} 根已结束交易周的周 K 线（进行中的交易周已剔除）"
            ),
        },
        "daily_indicators": daily_snapshot,
        "weekly_summary": weekly_snapshot,
        "key_levels": key_levels,
        "relative_strength": relative_strength
        or {"available": False, "benchmarks": [], "warnings": ["未获取基准数据"]},
    }
    return context


def build_relative_strength(
    asset_df: pd.DataFrame,
    benchmark_frames: Dict[str, pd.DataFrame],
    windows: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """基于交易日窗口计算个股相对各基准的表现。

    ``benchmark_frames`` 为 ``{基准代码: 日线 DataFrame}``。基准数据缺失时
    明确返回 ``available: False``，绝不虚构相对强弱结果。
    """
    from .indicators import RETURN_WINDOWS, relative_performance

    windows = windows or RETURN_WINDOWS
    result: Dict[str, object] = {"available": False, "benchmarks": [], "warnings": [], "window_note": ""}

    if asset_df is None or asset_df.empty:
        result["warnings"].append("个股数据为空，无法计算相对强弱")
        return result
    if not benchmark_frames:
        result["warnings"].append("未提供任何基准行情，无法计算相对强弱")
        return result

    asset = asset_df.copy()
    asset["Date"] = pd.to_datetime(asset["Date"])
    asset = asset.set_index("Date")

    any_available = False
    for bench_symbol, bench_df in benchmark_frames.items():
        if bench_df is None or bench_df.empty:
            result["warnings"].append(f"基准 {bench_symbol} 行情为空，已跳过")
            continue

        bench = bench_df.copy()
        bench["Date"] = pd.to_datetime(bench["Date"])
        bench = bench.set_index("Date")

        # 按日期对齐（仅保留双方都有数据的交易日），保证窗口口径一致
        joined = pd.concat(
            [asset["Close"].rename("asset"), bench["Close"].rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if len(joined) < 2:
            result["warnings"].append(f"基准 {bench_symbol} 与个股重叠的交易日不足，已跳过")
            continue

        perf = relative_performance(joined["asset"], joined["benchmark"], windows=windows)
        entry: Dict[str, object] = {
            "symbol": bench_symbol,
            "aligned_trading_days": int(len(joined)),
            "windows": perf.get("windows", {}),
            "as_of": _date_str(joined.index[-1]),
        }
        if perf.get("warnings"):
            entry["warnings"] = perf["warnings"]
            result["warnings"].extend(str(w) for w in perf["warnings"])  # type: ignore[union-attr]
        any_available = any_available or bool(perf.get("available"))
        result["benchmarks"].append(entry)  # type: ignore[union-attr]

    result["available"] = any_available
    result["window_note"] = "收益率窗口按交易日数量定义：1m=21, 3m=63, 6m=126 个交易日"
    if not any_available:
        result["warnings"].append("基准数据不可用，无法判断相对强弱")  # type: ignore[union-attr]
    return result
