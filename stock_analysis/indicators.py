"""技术指标计算模块。

设计说明
--------
* 只依赖 ``pandas`` / ``numpy``（项目已有依赖），不额外引入功能重复的 TA 库，
  从而避免不同库之间对同一指标的定义差异。
* 所有指标只使用**当前及之前**的数据（``rolling`` / ``ewm`` / 递推），
  不使用未来数据。
* 预热期不足时返回 ``NaN``，对外快照统一转换 ``None`` 并给出提示，
  **不会**用任何方式填充或伪造指标值。

指标算法口径
------------
====================  ==========================================================
指标                  定义
====================  ==========================================================
SMA(n)                ``close.rolling(n).mean()``
EMA(n)                ``close.ewm(span=n, adjust=False).mean()``（与主流看盘软件一致）
Wilder 平滑           经典 Wilder 递推：以最初 n 个值的**算术平均**为种子，
                      之后 ``avg = (avg*(n-1) + x) / n``（等价于 alpha = 1/n 的 RMA，
                      但使用 SMA 作为种子，与 Wilder 1978 原文一致）
RSI(14)               基于 Wilder 平滑的平均涨/跌幅；
                      ``RSI = 100 - 100/(1+RS)``，``RS = 平均涨幅 / 平均跌幅``；
                      边界定义：平均跌幅为 0 且平均涨幅 > 0 → 100；
                      两者均为 0（价格完全不变）→ 50；平均涨幅为 0 → 0
MACD(12,26,9)         ``DIF = EMA(12) - EMA(26)``；``DEA = EMA(DIF, 9)``；
                      **柱状图 Histogram = DIF - DEA**（注意：部分库使用
                      ``2*(DIF-DEA)``，本项目统一采用 ``DIF-DEA``）
ATR(14)               ``TR = max(H-L, |H-prevC|, |L-prevC|)``，再对 TR 做 Wilder 平滑
====================  ==========================================================

收益率窗口口径
--------------
以**交易日数量**定义：1 个月 = 21 个交易日，3 个月 = 63 个交易日，6 个月 = 126 个交易日。
收益率为 ``close[-1] / close[-1-n] - 1``。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 收益率窗口（交易日）
RETURN_WINDOWS: Dict[str, int] = {"1m": 21, "3m": 63, "6m": 126}

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_LENGTH = 14
ATR_LENGTH = 14
VOLUME_WINDOW = 20


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均（``adjust=False``，与主流看盘软件一致）。"""
    return series.ewm(span=span, adjust=False).mean()


def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    """Wilder 平滑（RMA）。

    以最初 ``length`` 个有效值的算术平均为种子，之后按
    ``avg = (avg * (length - 1) + x) / length`` 递推。前 ``length-1`` 个位置为 NaN。
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    n = len(values)
    out = np.full(n, np.nan)
    if n < length or length <= 0:
        return pd.Series(out, index=series.index, name=series.name)

    start = 0
    # 跳过开头的 NaN
    while start < n and np.isnan(values[start]):
        start += 1
    if n - start < length:
        return pd.Series(out, index=series.index, name=series.name)

    seed = float(np.mean(values[start : start + length]))
    out[start + length - 1] = seed
    prev = seed
    for i in range(start + length, n):
        current = values[i]
        if np.isnan(current):
            out[i] = np.nan
            continue
        prev = (prev * (length - 1) + current) / length
        out[i] = prev
    return pd.Series(out, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# 单项指标
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    """相对强弱指标（Wilder，见模块文档）。

    注意：``diff()`` 产生的第一个 NaN **不会被填充为 0**，而是由
    :func:`wilder_smooth` 跳过，这样种子正好是前 ``length`` 个**真实**涨幅/跌幅
    的平均值，与 Wilder 原文一致（首个有效值出现在第 ``length+1`` 根 K 线）。
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        result = 100.0 - (100.0 / (1.0 + rs))

    # 边界处理
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    only_loss_zero = (avg_loss == 0) & (avg_gain > 0)
    only_gain_zero = (avg_gain == 0) & (avg_loss > 0)

    result = result.mask(only_loss_zero, 100.0)
    result = result.mask(only_gain_zero, 0.0)
    result = result.mask(both_zero, 50.0)
    result = result.where(avg_gain.notna() & avg_loss.notna(), np.nan)
    result.name = f"RSI{length}"
    return result


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD。返回列 ``dif, dea, hist``，其中 ``hist = dif - dea``。"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = dif - dea
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 TR。

    ``TR = max(H-L, |H-prevC|, |L-prevC|)``。第一根 K 线没有前收盘价，
    因此按行业惯例取 ``H-L``（``DataFrame.max`` 会跳过 NaN）。
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = ATR_LENGTH) -> pd.Series:
    """平均真实波幅（Wilder 平滑）。"""
    return wilder_smooth(true_range(high, low, close), length)


def volume_stats(volume: pd.Series, window: int = VOLUME_WINDOW) -> pd.DataFrame:
    """成交量统计：当日量、N 日均量、量比。"""
    avg = volume.rolling(window=window, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = volume / avg
    return pd.DataFrame({"volume": volume, "volume_avg": avg, "volume_ratio": ratio})


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """N 个交易日的前后收益率（``close / close.shift(window) - 1``）。"""
    return close / close.shift(window) - 1.0


# ---------------------------------------------------------------------------
# 组合计算
# ---------------------------------------------------------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为 OHLCV DataFrame 追加全部指标列（返回新对象）。"""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce")

    for window in (20, 50, 200):
        out[f"MA{window}"] = sma(close, window)

    m = macd(close)
    out["MACD_DIF"] = m["dif"]
    out["MACD_DEA"] = m["dea"]
    out["MACD_HIST"] = m["hist"]
    out["RSI14"] = rsi(close)
    out["ATR14"] = atr(high, low, close)
    out["VOL_AVG20"] = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    out["VOL_RATIO20"] = volume / out["VOL_AVG20"].replace(0, np.nan)

    for name, window in RETURN_WINDOWS.items():
        out[f"RET_{name}"] = rolling_return(close, window)

    return out


def _clean_number(value: object, digits: int = 4) -> Optional[float]:
    """把 numpy/pandas 数值安全地转换为 ``Optional[float]``。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def _pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return round((numerator / denominator - 1.0) * 100.0, 2)


def classify_trend(
    close: float,
    ma_fast: Optional[float],
    ma_slow: Optional[float],
    slow_slope_up: Optional[bool],
    fast_label: str,
    slow_label: str,
) -> Tuple[str, List[str]]:
    """基于 ``收盘价 vs 均线`` 与 ``均线斜率`` 的可解释趋势判定。

    规则（全部满足才算对应趋势，否则视为震荡）：

    * 上升：收盘价 > 慢线，快线 > 慢线，且慢线向上
    * 下降：收盘价 < 慢线，快线 < 慢线，且慢线向下
    * 其余：震荡

    返回 ``(趋势中文标签, 依据列表)``。
    """
    evidence: List[str] = []
    if ma_fast is None or ma_slow is None:
        return "数据不足", [f"缺少 {fast_label} 或 {slow_label}，无法判定趋势"]

    above_slow = close > ma_slow
    fast_above_slow = ma_fast > ma_slow
    evidence.append(
        "收盘价 {:.2f} {} {} {:.2f}".format(close, "高于" if above_slow else "低于", slow_label, ma_slow)
    )
    evidence.append(
        "{} {:.2f} {} {} {:.2f}".format(
            fast_label, ma_fast, "高于" if fast_above_slow else "低于", slow_label, ma_slow
        )
    )
    if slow_slope_up is None:
        evidence.append(f"{slow_label} 斜率不可用")
    else:
        evidence.append(f"{slow_label} 近期{'向上' if slow_slope_up else '走平或向下'}")

    if above_slow and fast_above_slow and slow_slope_up is True:
        return "上升", evidence
    if (not above_slow) and (not fast_above_slow) and slow_slope_up is False:
        return "下降", evidence
    return "震荡", evidence


def macd_state(hist: Optional[float], hist_prev: Optional[float], dif: Optional[float], dea: Optional[float]) -> Dict[str, object]:
    """MACD 状态描述：金叉/死叉、柱状图变化、多空。"""
    state: Dict[str, object] = {"cross": "none", "hist_trend": "unknown", "bias": "unknown"}
    if hist is None or dif is None or dea is None:
        return state
    state["bias"] = "多头" if dif > dea else "空头"
    if hist_prev is not None:
        if hist_prev <= 0 < hist:
            state["cross"] = "golden"  # 金叉
        elif hist_prev >= 0 > hist:
            state["cross"] = "death"  # 死叉
        if hist > hist_prev:
            state["hist_trend"] = "增强"
        elif hist < hist_prev:
            state["hist_trend"] = "减弱"
        else:
            state["hist_trend"] = "持平"
    return state


def rsi_state(value: Optional[float]) -> str:
    """RSI 区间描述。"""
    if value is None:
        return "数据不足"
    if value >= 70:
        return "超买区(>=70)"
    if value >= 60:
        return "偏强(60-70)"
    if value > 40:
        return "中性(40-60)"
    if value > 30:
        return "偏弱(30-40)"
    return "超卖区(<=30)"


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------


def _slope_up(series: pd.Series, lookback: int) -> Optional[bool]:
    """判断某均线在最近 ``lookback`` 根 K 线中是否向上。"""
    if series is None or len(series) <= lookback:
        return None
    last = series.iloc[-1]
    prev = series.iloc[-1 - lookback]
    if pd.isna(last) or pd.isna(prev):
        return None
    return bool(last > prev)


def daily_indicator_snapshot(df: pd.DataFrame, slope_lookback: int = 20) -> Dict[str, object]:
    """计算日线指标快照（仅使用最后一根 K 线及其历史）。

    ``df`` 需要包含标准 OHLCV 列；本函数会自动追加指标列。
    """
    if df is None or df.empty:
        return {"available": False, "warnings": ["行情数据为空，无法计算日线指标"]}

    data = add_indicators(df)
    last = data.iloc[-1]
    warnings: List[str] = []

    close = _clean_number(last["Close"], 4)
    snapshot: Dict[str, object] = {
        "available": True,
        "as_of": pd.Timestamp(last["Date"]).strftime("%Y-%m-%d"),
        "close": close,
        "ma": {},
        "price_vs_ma_pct": {},
        "rsi": {},
        "macd": {},
        "atr": {},
        "volume": {},
        "returns_pct": {},
        "trend": {},
        "warnings": warnings,
    }

    # 均线
    for window in (20, 50, 200):
        value = _clean_number(last.get(f"MA{window}"), 4)
        snapshot["ma"][f"ma{window}"] = value
        if value is None:
            warnings.append(
                f"MA{window} 不可用（历史数据不足 {window} 根 K 线，当前 {len(data)} 根，属正常预热期）"
            )
        snapshot["price_vs_ma_pct"][f"vs_ma{window}"] = _pct(close, value)

    ma20 = snapshot["ma"]["ma20"]
    ma50 = snapshot["ma"]["ma50"]
    ma200 = snapshot["ma"]["ma200"]

    # MACD
    snapshot["macd"] = {
        "dif": _clean_number(last.get("MACD_DIF"), 4),
        "dea": _clean_number(last.get("MACD_DEA"), 4),
        "hist": _clean_number(last.get("MACD_HIST"), 4),
        "hist_prev": _clean_number(data["MACD_HIST"].iloc[-2], 4) if len(data) >= 2 else None,
    }
    snapshot["macd"].update(
        macd_state(
            snapshot["macd"]["hist"],
            snapshot["macd"]["hist_prev"],
            snapshot["macd"]["dif"],
            snapshot["macd"]["dea"],
        )
    )

    # RSI
    rsi_value = _clean_number(last.get("RSI14"), 2)
    snapshot["rsi"] = {"rsi14": rsi_value, "state": rsi_state(rsi_value)}
    if rsi_value is None:
        warnings.append("RSI14 不可用（历史数据不足 15 根 K 线）")

    # ATR
    atr_value = _clean_number(last.get("ATR14"), 4)
    atr_pct = None
    if atr_value is not None and close:
        atr_pct = round(atr_value / close * 100.0, 2)
    snapshot["atr"] = {"atr14": atr_value, "atr_pct": atr_pct}
    if atr_value is None:
        warnings.append("ATR14 不可用（历史数据不足 15 根 K 线）")

    # 成交量
    volume_last = _clean_number(last.get("Volume"), 2)
    volume_avg = _clean_number(last.get("VOL_AVG20"), 2)
    volume_ratio = _clean_number(last.get("VOL_RATIO20"), 4)
    snapshot["volume"] = {
        "volume": volume_last,
        "volume_avg20": volume_avg,
        "volume_ratio20": volume_ratio,
    }
    if volume_ratio is None:
        warnings.append("20 日均量不可用（历史数据不足 20 根 K 线）")

    # 收益率
    for name, window in RETURN_WINDOWS.items():
        value = _clean_number(last.get(f"RET_{name}"), 4)
        pct = None if value is None else round(value * 100.0, 2)
        snapshot["returns_pct"][name] = {"trading_days": window, "value": pct}
        if value is None:
            warnings.append(f"近 {name}（{window} 个交易日）收益率不可用：历史数据不足")

    # 趋势
    trend, evidence = classify_trend(
        close=close if close is not None else float("nan"),
        ma_fast=ma20,
        ma_slow=ma50,
        slow_slope_up=_slope_up(data["MA50"], slope_lookback),
        fast_label="MA20",
        slow_label="MA50",
    )
    long_trend_evidence = []
    if ma200 is not None and close is not None:
        long_trend_evidence.append(
            "收盘价 {} MA200 {:.2f}（长期均线）".format("高于" if close > ma200 else "低于", ma200)
        )
    snapshot["trend"] = {
        "trend": trend,
        "evidence": evidence,
        "ma200_context": long_trend_evidence,
    }

    return snapshot


def weekly_summary(
    weekly: pd.DataFrame,
    ma_fast_window: int = 10,
    ma_slow_window: int = 30,
    slope_lookback: int = 4,
    return_windows: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """周线趋势摘要。

    ``weekly`` 应当是 :func:`stock_analysis.kline.aggregate_weekly` 的输出
    （默认已剔除进行中的交易周）。
    """
    if weekly is None or weekly.empty:
        return {"available": False, "warnings": ["周线数据为空"]}

    data = weekly.copy()
    close = pd.to_numeric(data["Close"], errors="coerce")
    data["MA_FAST"] = sma(close, ma_fast_window)
    data["MA_SLOW"] = sma(close, ma_slow_window)
    data["RSI14"] = rsi(close)
    m = macd(close)
    data["MACD_DIF"] = m["dif"]
    data["MACD_DEA"] = m["dea"]
    data["MACD_HIST"] = m["hist"]

    windows = return_windows or {"4w": 4, "13w": 13, "26w": 26}
    for name, window in windows.items():
        data[f"RET_{name}"] = rolling_return(close, window)

    last = data.iloc[-1]
    warnings: List[str] = []

    close_value = _clean_number(last["Close"])
    ma_fast = _clean_number(last["MA_FAST"])
    ma_slow = _clean_number(last["MA_SLOW"])
    if ma_fast is None:
        warnings.append(f"周线 MA{ma_fast_window} 不可用（周线根数不足 {ma_fast_window}）")
    if ma_slow is None:
        warnings.append(f"周线 MA{ma_slow_window} 不可用（周线根数不足 {ma_slow_window}）")

    trend, evidence = classify_trend(
        close=close_value if close_value is not None else float("nan"),
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        slow_slope_up=_slope_up(data["MA_SLOW"], slope_lookback),
        fast_label=f"MA{ma_fast_window}",
        slow_label=f"MA{ma_slow_window}",
    )

    rsi_value = _clean_number(last["RSI14"], 2)

    # 近 N 周的高低点结构（用于判断“更高的高点 / 更低的低点”）
    structure: List[str] = []
    if len(data) >= 6:
        recent = data.tail(4)
        prior = data.iloc[-8:-4] if len(data) >= 8 else data.iloc[:-4]
        if len(prior) >= 2:
            if recent["High"].max() > prior["High"].max():
                structure.append("近 4 周高点高于此前 4 周高点（更高的高点）")
            else:
                structure.append("近 4 周高点未超过此前 4 周高点")
            if recent["Low"].min() > prior["Low"].min():
                structure.append("近 4 周低点高于此前 4 周低点（更高的低点）")
            else:
                structure.append("近 4 周低点未高于此前 4 周低点")

    summary: Dict[str, object] = {
        "available": True,
        "as_of": pd.Timestamp(last["Date"]).strftime("%Y-%m-%d"),
        "weeks": int(len(data)),
        "close": close_value,
        "ma_fast": {f"ma{ma_fast_window}": ma_fast},
        "ma_slow": {f"ma{ma_slow_window}": ma_slow},
        "price_vs_ma_slow_pct": _pct(close_value, ma_slow),
        "rsi14": rsi_value,
        "rsi_state": rsi_state(rsi_value),
        "macd": {
            "dif": _clean_number(last["MACD_DIF"]),
            "dea": _clean_number(last["MACD_DEA"]),
            "hist": _clean_number(last["MACD_HIST"]),
            "hist_prev": _clean_number(data["MACD_HIST"].iloc[-2]) if len(data) >= 2 else None,
        },
        "trend": trend,
        "evidence": evidence,
        "structure": structure,
        "returns_pct": {
            name: {
                "weeks": window,
                "value": (
                    None
                    if _clean_number(last.get(f"RET_{name}")) is None
                    else round(float(last[f"RET_{name}"]) * 100.0, 2)
                ),
            }
            for name, window in windows.items()
        },
        "warnings": warnings,
    }
    summary["macd"].update(  # type: ignore[union-attr]
        macd_state(
            summary["macd"]["hist"],  # type: ignore[index]
            summary["macd"]["hist_prev"],  # type: ignore[index]
            summary["macd"]["dif"],  # type: ignore[index]
            summary["macd"]["dea"],  # type: ignore[index]
        )
    )
    return summary


def relative_performance(
    asset_close: pd.Series,
    benchmark_close: pd.Series,
    windows: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """计算个股相对基准的超额收益。

    ``asset_close`` / ``benchmark_close`` 需要是**按日期对齐**的收盘价序列
    （索引为日期）。窗口以交易日数量定义，双方使用完全相同的窗口。
    """
    windows = windows or RETURN_WINDOWS
    asset = pd.to_numeric(pd.Series(asset_close), errors="coerce")
    bench = pd.to_numeric(pd.Series(benchmark_close), errors="coerce")

    result: Dict[str, object] = {"available": False, "windows": {}, "warnings": []}
    if len(asset) < 2 or len(bench) < 2:
        result["warnings"].append("个股或基准数据不足，无法计算相对强弱")
        return result

    available = 0
    for name, window in windows.items():
        if len(asset) <= window or len(bench) <= window:
            result["windows"][name] = {"trading_days": window, "asset": None, "benchmark": None, "excess": None}
            result["warnings"].append(f"近 {name}（{window} 个交易日）数据不足，无法对比")
            continue
        asset_ret = float(asset.iloc[-1] / asset.iloc[-1 - window] - 1.0)
        bench_ret = float(bench.iloc[-1] / bench.iloc[-1 - window] - 1.0)
        result["windows"][name] = {
            "trading_days": window,
            "asset": round(asset_ret * 100.0, 2),
            "benchmark": round(bench_ret * 100.0, 2),
            "excess": round((asset_ret - bench_ret) * 100.0, 2),
        }
        available += 1

    result["available"] = available > 0
    if not result["available"]:
        result["warnings"].append("所有窗口的基准数据都不足，未计算相对强弱")
    return result
