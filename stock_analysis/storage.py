"""分析记录持久化与后续复核。

存储形式
--------
不使用数据库，采用文件系统（JSON + Markdown），便于后续扩展：

``reports/<SYMBOL>/<分析日期>_<时间戳>.json``
    完整结构化记录：行情元信息、指标摘要、关键价位、数据质量、AI 结构化结果、
    模型名称与 Prompt 版本、以及预留的 ``forward_returns``（用于后续评估）。
``reports/<SYMBOL>/<分析日期>_<时间戳>.md``
    便于阅读的中文 Markdown 报告。

后续复核（``forward_returns``）
------------------------------
:func:`review_records` 会在分析日过去 ``min_age_days`` 天之后，**重新抓取**
该标的在此之后的行情，计算未来 5/10/20 个交易日的收益率、最大有利波动（MFE）
与最大不利波动（MAE）。

避免未来数据泄漏的做法：
* 未来收益只使用 ``analysis_date`` **之后**的 K 线；
* 收益基准统一使用"复核时重新抓取"的当日收盘价（与未来价格同一复权口径），
  而不是历史记录里可能因复权因子变化而失真的旧价格。
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RECORD_VERSION = 1
FORWARD_HORIZONS = (5, 10, 20)


def safe_name(value: str) -> str:
    """把标的代码转换为安全的文件名片段。"""
    return re.sub(r'[\\/*?:"<>|=]', "_", str(value))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def record_dir(reports_dir: str, symbol: str) -> str:
    return os.path.join(reports_dir, safe_name(symbol))


def save_analysis(record: Dict[str, Any], reports_dir: str) -> Dict[str, str]:
    """保存分析记录（JSON）与报告（Markdown），返回文件路径。"""
    symbol = str(record.get("symbol", "UNKNOWN"))
    directory = ensure_dir(record_dir(reports_dir, symbol))

    analysis_date = str(record.get("analysis_date") or datetime.now().strftime("%Y-%m-%d"))
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    base = f"{analysis_date}_{stamp}"

    json_path = os.path.join(directory, f"{base}.json")
    md_path = os.path.join(directory, f"{base}.md")

    payload = dict(record)
    payload["record_version"] = RECORD_VERSION
    payload.setdefault("forward_returns", None)
    payload["saved_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

    markdown = record.get("markdown")
    if markdown:
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)

    logger.info("分析记录已保存：%s", json_path)
    return {"json": json_path, "markdown": md_path}


def iter_record_paths(reports_dir: str) -> List[str]:
    """列出所有分析记录 JSON 文件（按时间倒序无所谓，调用方可自行排序）。"""
    if not os.path.isdir(reports_dir):
        return []
    paths: List[str] = []
    for root, _dirs, files in os.walk(reports_dir):
        for name in files:
            if name.endswith(".json"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def load_record(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_record_in_place(path: str, record: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 未来收益复核
# ---------------------------------------------------------------------------


def _clean(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def evaluate_forward_returns(
    record: Dict[str, Any],
    forward_daily: pd.DataFrame,
    horizons: Tuple[int, ...] = FORWARD_HORIZONS,
) -> Dict[str, Any]:
    """根据分析日之后的行情，计算未来收益与 MFE/MAE。

    ``forward_daily``：包含 ``analysis_date`` 之前数据的日线（函数内部会自行
    过滤出**严格晚于**分析日的 K 线）。返回值写入 ``record['forward_returns']``。
    """
    analysis_date = pd.Timestamp(str(record.get("analysis_date"))).normalize()
    stored_close = _clean(record.get("close_price"))

    result: Dict[str, Any] = {
        "analysis_date": analysis_date.strftime("%Y-%m-%d"),
        "horizons": {},
        "complete": False,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if forward_daily is None or forward_daily.empty:
        result["error"] = "复核用行情为空"
        return result

    df = forward_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # 基准价：优先使用同一批数据中分析日的收盘价（口径一致，避免复权因子漂移）
    same_day = df.loc[df["Date"] == analysis_date]
    if not same_day.empty:
        base_close = _clean(same_day["Close"].iloc[-1])
        result["basis"] = "复核数据中分析日收盘价"
    else:
        base_close = stored_close
        result["basis"] = "记录中的历史收盘价（复核数据未包含分析日）"

    result["base_close"] = base_close
    if base_close in (None, 0):
        result["error"] = "缺少可用的基准收盘价，无法计算未来收益"
        return result

    future = df.loc[df["Date"] > analysis_date].reset_index(drop=True)
    result["future_bars"] = int(len(future))
    if future.empty:
        result["error"] = "分析日之后暂无新的交易日数据"
        return result

    max_horizon = max(horizons)
    window = future.head(max_horizon)

    for horizon in horizons:
        if len(future) >= horizon:
            future_close = _clean(future["Close"].iloc[horizon - 1])
            ret = None if future_close is None else round(future_close / base_close - 1.0, 4)
            result["horizons"][f"{horizon}d"] = {
                "trading_days": horizon,
                "date": pd.Timestamp(future["Date"].iloc[horizon - 1]).strftime("%Y-%m-%d"),
                "return": ret,
            }
        else:
            result["horizons"][f"{horizon}d"] = {
                "trading_days": horizon,
                "date": None,
                "return": None,
                "note": f"仅有 {len(future)} 个后续交易日，尚不足 {horizon} 个",
            }

    high = _clean(window["High"].max())
    low = _clean(window["Low"].min())
    result["max_favorable_excursion"] = None if high is None else round(high / base_close - 1.0, 4)
    result["max_adverse_excursion"] = None if low is None else round(low / base_close - 1.0, 4)
    result["window_bars"] = int(len(window))
    result["complete"] = int(len(future)) >= max_horizon
    if not result["complete"]:
        result["note"] = (
            f"当前仅积累了 {len(future)} 个后续交易日，"
            f"不足 {max_horizon} 日完整观察窗口，可稍后重新复核"
        )
    return result


def review_records(
    reports_dir: str,
    min_age_days: int = 30,
    max_horizon: int = 20,
    fetch_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    force: bool = False,
    reference_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """对历史分析记录做未来收益复核。

    ``fetch_fn(symbol) -> 日线 DataFrame``，默认使用 :mod:`stock_analysis.kline` 抓取
    最近 1 年日线（覆盖最长 20 个交易日的观察窗口）。测试中可注入。
    """
    if fetch_fn is None:
        fetch_fn = _default_fetch

    reference = pd.Timestamp(reference_date or datetime.now()).normalize()
    updates: List[Dict[str, Any]] = []

    for path in iter_record_paths(reports_dir):
        try:
            record = load_record(path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("跳过无法读取的记录 %s：%s", path, exc)
            continue

        existing = record.get("forward_returns")
        if existing and existing.get("complete") and not force:
            continue

        analysis_date = record.get("analysis_date")
        symbol = record.get("symbol")
        if not analysis_date or not symbol:
            continue

        age_days = int((reference - pd.Timestamp(str(analysis_date)).normalize()).days)
        if age_days < min_age_days and not force:
            continue

        try:
            daily = fetch_fn(str(symbol))
        except Exception as exc:  # 网络或数据源问题不应中断整个复核流程
            logger.warning("复核 %s 时获取行情失败：%s", symbol, exc)
            continue

        forward = evaluate_forward_returns(
            record, daily, horizons=tuple(sorted(set(FORWARD_HORIZONS) | {max_horizon}))
        )
        record["forward_returns"] = forward
        save_record_in_place(path, record)
        updates.append({"symbol": symbol, "path": path, "forward_returns": forward})
        logger.info("已复核 %s（分析日 %s）", symbol, analysis_date)

    return updates


def _default_fetch(symbol: str) -> pd.DataFrame:
    """默认复核取数：抓取近 1 年日线。"""
    from . import kline

    df, _quality = kline.fetch_daily(symbol, period="1y")
    return df
