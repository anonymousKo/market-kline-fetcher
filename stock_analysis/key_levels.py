"""关键价位与近期价格结构。

全部基于**可解释的规则**计算，不使用黑箱模型，也不让大模型凭空编造价位：

* 最近 N 个交易日的最高价 / 最低价（N = 20 / 60）。
* 局部高低点（分形枢轴，Fractal pivot）：``span`` 根 K 线内的最高/最低点。
  由于需要 ``span`` 根**之后**的 K 线才能确认，最近 ``span`` 根 K 线内的枢轴
  不会被使用（避免使用未确认/未来信息）。
* 支撑 / 阻力**区域**：把价格相近的枢轴聚类成区间。
  聚类容差 = ``max(0.5 * ATR14, 1% * 当前收盘价)``，在容差内的枢轴归为同一区域。
  区域中心价 = 组内枢轴价格均值；区域范围 = 组内最低价 ~ 最高价。
* 突破位置：若当前收盘价高于"前 20 个交易日（剔除最近 5 日）的最高点"，
  则该突破位被视为潜在的支撑参考。
* 价格与 MA20 / MA50 / MA200 的距离。

当枢轴数量不足或不存在有效水平时，明确输出"暂无足够证据"，
而不是给出不可靠的精确价位。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import ATR_LENGTH, atr as compute_atr

logger = logging.getLogger(__name__)


def recent_extremes(df: pd.DataFrame, windows: Tuple[int, ...] = (20, 60)) -> Dict[str, object]:
    """最近 N 个交易日区间的最高价 / 最低价及其日期。"""
    result: Dict[str, object] = {}
    for window in windows:
        if len(df) < 1:
            continue
        subset = df.tail(window)
        if subset.empty:
            continue
        high_idx = subset["High"].idxmax()
        low_idx = subset["Low"].idxmin()
        result[f"last{window}d"] = {
            "window": window,
            "bars_used": int(len(subset)),
            "high": _num(subset.loc[high_idx, "High"]),
            "high_date": _date(subset.loc[high_idx, "Date"]),
            "low": _num(subset.loc[low_idx, "Low"]),
            "low_date": _date(subset.loc[low_idx, "Date"]),
        }
    return result


def find_pivots(df: pd.DataFrame, span: int = 2) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """用分形法找出局部高点与局部低点。

    只有前后各 ``span`` 根 K 线都低于（高于）该点的极值才认定为枢轴，
    因此最近 ``span`` 根 K 线内的潜在枢轴不会被确认，也就不会被使用。
    """
    highs: List[Dict[str, object]] = []
    lows: List[Dict[str, object]] = []
    if df is None or len(df) < 2 * span + 1:
        return highs, lows

    window = 2 * span + 1
    high_series = pd.to_numeric(df["High"], errors="coerce")
    low_series = pd.to_numeric(df["Low"], errors="coerce")
    dates = pd.to_datetime(df["Date"])

    rolling_high = high_series.rolling(window=window, center=True, min_periods=window).max()
    rolling_low = low_series.rolling(window=window, center=True, min_periods=window).min()

    for i in range(len(df)):
        value = high_series.iloc[i]
        if pd.notna(value) and pd.notna(rolling_high.iloc[i]) and value >= rolling_high.iloc[i]:
            highs.append({"date": _date(dates.iloc[i]), "price": _num(value)})
        value = low_series.iloc[i]
        if pd.notna(value) and pd.notna(rolling_low.iloc[i]) and value <= rolling_low.iloc[i]:
            lows.append({"date": _date(dates.iloc[i]), "price": _num(value)})
    return highs, lows


def cluster_zones(
    pivots: List[Dict[str, object]],
    tolerance: float,
    last_date: Optional[str] = None,
) -> List[Dict[str, object]]:
    """把价格相近的枢轴聚合成支撑/阻力区域。"""
    prices = [p for p in pivots if p.get("price") is not None]
    if not prices or tolerance <= 0:
        return []

    ordered = sorted(prices, key=lambda item: float(item["price"]))
    clusters: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = [ordered[0]]

    for pivot in ordered[1:]:
        if float(pivot["price"]) - float(current[0]["price"]) <= tolerance:
            current.append(pivot)
        else:
            clusters.append(current)
            current = [pivot]
    clusters.append(current)

    zones: List[Dict[str, object]] = []
    for cluster in clusters:
        members = sorted(cluster, key=lambda item: str(item["date"]))
        values = [float(item["price"]) for item in cluster]
        # 只保留最近 3 次触及，避免噪音
        recent = members[-3:]
        zones.append(
            {
                "level": round(float(np.mean(values)), 4),
                "range": [round(min(values), 4), round(max(values), 4)],
                "touches": len(cluster),
                "last_date": str(recent[-1]["date"]),
                "dates": [str(item["date"]) for item in recent],
            }
        )
    return zones


def build_key_levels(
    df: pd.DataFrame,
    pivot_span: int = 2,
    atr_multiplier: float = 0.5,
    min_tolerance_pct: float = 0.01,
    max_pivots_per_side: int = 6,
) -> Dict[str, object]:
    """生成关键价位与价格结构结果（供报告与 AI 上下文使用）。"""
    if df is None or df.empty:
        return {
            "available": False,
            "notes": ["行情数据为空，无法识别关键价位"],
            "insufficient_evidence": True,
        }

    close = _num(df["Close"].iloc[-1])
    as_of = _date(df["Date"].iloc[-1])
    notes: List[str] = []

    data = df.copy()
    data["ATR14"] = compute_atr(data["High"], data["Low"], data["Close"], ATR_LENGTH)
    atr_value = _num(data["ATR14"].iloc[-1])

    tolerance = 0.0
    if close is not None:
        tolerance = max(
            (atr_value or 0.0) * atr_multiplier,
            abs(close) * min_tolerance_pct,
        )

    pivot_highs, pivot_lows = find_pivots(data, span=pivot_span)
    if len(pivot_highs) > max_pivots_per_side:
        pivot_highs = pivot_highs[-max_pivots_per_side:]
    if len(pivot_lows) > max_pivots_per_side:
        pivot_lows = pivot_lows[-max_pivots_per_side:]

    high_zones = cluster_zones(pivot_highs, tolerance)
    low_zones = cluster_zones(pivot_lows, tolerance)

    resistance_zones: List[Dict[str, object]] = []
    support_zones: List[Dict[str, object]] = []
    if close is not None:
        for zone in high_zones:
            zone["basis"] = "局部高点（分形枢轴）聚类"
            (resistance_zones if zone["level"] > close else support_zones).append(zone)
        for zone in low_zones:
            zone["basis"] = "局部低点（分形枢轴）聚类"
            (resistance_zones if zone["level"] > close else support_zones).append(zone)
    else:
        for zone in high_zones:
            zone["basis"] = "局部高点（分形枢轴）聚类"
            resistance_zones.append(zone)
        for zone in low_zones:
            zone["basis"] = "局部低点（分形枢轴）聚类"
            support_zones.append(zone)

    resistance_zones.sort(key=lambda z: float(z["level"]))
    support_zones.sort(key=lambda z: float(z["level"]), reverse=True)

    # 突破位（前 20 日高点，剔除最近 5 日，避免把当前行情自身算进去）
    breakout_notes: List[str] = []
    if len(data) > 25 and close is not None:
        prior_window = data.iloc[-25:-5]
        prior_high = _num(prior_window["High"].max())
        prior_low = _num(prior_window["Low"].min())
        if prior_high is not None and close > prior_high:
            breakout_notes.append(
                f"当前收盘价 {close:.2f} 已站上前 20 日高点 {prior_high:.2f}"
                f"（{_date(prior_window['Date'].iloc[int(prior_window['High'].argmax())])}），"
                "该突破位可作为潜在支撑参考"
            )
        if prior_low is not None and close < prior_low:
            breakout_notes.append(
                f"当前收盘价 {close:.2f} 已跌破前 20 日低点 {prior_low:.2f}，该位置可作为潜在阻力参考"
            )

    # 与均线的距离
    ma_distances: Dict[str, Optional[float]] = {}
    for window in (20, 50, 200):
        if len(data) >= window:
            ma_value = _num(pd.to_numeric(data["Close"], errors="coerce").rolling(window).mean().iloc[-1])
            if ma_value and close is not None:
                ma_distances[f"ma{window}"] = round((close / ma_value - 1.0) * 100.0, 2)
            else:
                ma_distances[f"ma{window}"] = None
        else:
            ma_distances[f"ma{window}"] = None

    insufficient = not resistance_zones and not support_zones
    if insufficient:
        notes.append(
            "枢轴数量不足，暂无足够证据通过聚类识别可靠的支撑/阻力区域；"
            "仅提供近期区间高低点作为参考"
        )
    if resistance_zones:
        for zone in resistance_zones[:3]:
            notes.append(
                "阻力区 {:.2f}（区间 {:.2f}~{:.2f}，{} 次触及，最近 {})".format(
                    zone["level"], zone["range"][0], zone["range"][1], zone["touches"], zone["last_date"]
                )
            )
    if support_zones:
        for zone in support_zones[:3]:
            notes.append(
                "支撑区 {:.2f}（区间 {:.2f}~{:.2f}，{} 次触及，最近 {})".format(
                    zone["level"], zone["range"][0], zone["range"][1], zone["touches"], zone["last_date"]
                )
            )

    return {
        "available": True,
        "as_of": as_of,
        "close": close,
        "atr14": atr_value,
        "cluster_tolerance": round(tolerance, 4),
        "cluster_rule": (
            f"枢轴价格差 <= max({atr_multiplier}*ATR14, {min_tolerance_pct:.0%}*收盘价) 视为同一区域"
        ),
        "recent_extremes": recent_extremes(data),
        "nearest_resistance": resistance_zones[0] if resistance_zones else None,
        "nearest_support": support_zones[0] if support_zones else None,
        "resistance_zones": resistance_zones[:5],
        "support_zones": support_zones[:5],
        "pivot_highs": pivot_highs[-5:],
        "pivot_lows": pivot_lows[-5:],
        "breakout_notes": breakout_notes,
        "ma_distances_pct": ma_distances,
        "notes": notes,
        "insufficient_evidence": insufficient,
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _num(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, 4)


def _date(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None
