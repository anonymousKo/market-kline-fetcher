"""行情数据层：获取、规范化、数据质量检查、日线聚合周线。

该模块复用并统一了原项目 ``fetch_data.py`` 的行情获取方式（yfinance），
但不改变其对外行为；``fetch_data.py`` 现在通过本模块取数，保证口径一致。

关键约定
--------
* 规范列名固定为 ``Date, Open, High, Low, Close, Volume``。
* ``Date`` 为交易日的本地日期（``datetime64[ns]``，无时区），向下游统一输出
  ``YYYY-MM-DD`` 字符串。
* 复权口径固定为 ``auto_adjust=True``（后复权调整后的 OHLC），避免把未复权价格
  和复权价格混用。周线由**日线**聚合得到，因此两者口径一致。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# yfinance 可能返回的各种列名 -> 规范列名
_COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "AdjClose",
    "adjclose": "AdjClose",
    "volume": "Volume",
    # yfinance 在某些周期会额外返回的列，直接忽略
    "capital gains": "CapitalGains",
    "dividends": "Dividends",
    "stock splits": "StockSplits",
}

# 默认复权口径说明（用于报告与记录）
ADJUSTMENT_NOTE = "auto_adjust=True（yfinance 后复权调整后的 OHLC，含分红/拆股调整）"

# 各市场的交易所时区（用于判断“当日交易是否已经收盘”）
_MARKET_TIMEZONES: Dict[str, str] = {
    "us_equity": "America/New_York",
    "us_etf": "America/New_York",
    "index": "America/New_York",
    "futures": "America/New_York",
    "forex": "UTC",
    "china_a_shanghai": "Asia/Shanghai",
    "china_a_shenzhen": "Asia/Shanghai",
    "hong_kong": "Asia/Hong_Kong",
    "japan": "Asia/Tokyo",
    "uk": "Europe/London",
    "canada": "America/Toronto",
    "unknown": "UTC",
}

# 收盘后等待多少分钟才认为当日数据已经结算完毕（数据源仍可能返回盘中值）
DEFAULT_SETTLE_MINUTES = 30


def infer_timezone(symbol: str) -> str:
    """根据代码推断**交易所所在时区**，用于判断当日交易是否已收盘。

    无法识别时返回 ``UTC``，不会猜测成具体市场。
    """
    market, _currency = infer_market(symbol)
    return _MARKET_TIMEZONES.get(market, "UTC")


def market_now(timezone_name: str) -> pd.Timestamp:
    """返回指定时区的当前时间（时区感知）。时区名非法时退化为 ``UTC``。"""
    try:
        return pd.Timestamp.now(tz=timezone_name)
    except Exception:  # pragma: no cover - 非法时区名兜底
        logger.warning("无法识别的时区 %r，已退化为 UTC", timezone_name)
        return pd.Timestamp.now(tz="UTC")


def session_in_progress(
    last_bar_date: pd.Timestamp,
    timezone_name: str,
    close_time: str = "16:00",
    settle_minutes: int = DEFAULT_SETTLE_MINUTES,
    now: Optional[pd.Timestamp] = None,
) -> Tuple[bool, Optional[str]]:
    """判断最后一根日 K 线是否属于**尚未结束**的交易日。

    返回 ``(是否未结束, 原因说明)``。判定逻辑（全部基于**交易所本地时间**，
    而不是运行脚本的机器时区）：

    1. 若最后一根 K 线的日期早于交易所当地今天 -> 已收盘，返回 ``(False, None)``；
    2. 若等于当地今天，且当地现在时间早于收盘时间 -> 未结束；
    3. 若处于收盘时间后 ``settle_minutes`` 分钟内 -> 数据源可能仍在结算，保守视为未结束；
    4. 否则视为已收盘（周末同理）。

    ``now`` 可注入，便于测试与“指定分析时间”的场景。
    """
    reference = now if now is not None else market_now(timezone_name)
    if reference.tzinfo is None:  # 允许传入 naive 时间（按市场时区解释）
        reference = reference.tz_localize(timezone_name)

    local = reference.tz_convert(timezone_name)
    local_date = local.normalize().tz_localize(None)
    bar_date = pd.Timestamp(last_bar_date).normalize()

    if bar_date < local_date:
        return False, None
    if bar_date > local_date:
        # 数据日期晚于当地今天，属于异常数据，不做“盘中”判断
        return False, None
    if local.weekday() >= 5:
        return False, None

    try:
        hour, minute = (int(part) for part in str(close_time).split(":", 1))
    except (ValueError, TypeError):
        hour, minute = 16, 0
    close_ts = local.normalize() + pd.Timedelta(hours=hour, minutes=minute)

    if local < close_ts:
        return True, (
            "当日交易尚未收盘（交易所当地时间 {}，收盘时间 {}）".format(
                local.strftime("%Y-%m-%d %H:%M"), close_ts.strftime("%H:%M")
            )
        )
    if local < close_ts + pd.Timedelta(minutes=max(0, settle_minutes)):
        return True, (
            "刚过收盘时间（交易所当地时间 {}），当日行情可能仍在结算".format(
                local.strftime("%Y-%m-%d %H:%M")
            )
        )
    return False, None


# ---------------------------------------------------------------------------
# 市场识别
# ---------------------------------------------------------------------------


def infer_market(symbol: str) -> Tuple[str, str]:
    """根据代码后缀粗略推断市场与币种。

    仅用于报告展示与记录，不影响取数逻辑。无法判断时返回 ``unknown``，
    不会虚构具体市场信息。
    """
    s = symbol.strip().upper()
    if s.endswith("=F"):
        return "futures", "USD"
    if s.endswith("=X"):
        return "forex", "USD"
    if s.endswith(".SS"):
        return "china_a_shanghai", "CNY"
    if s.endswith(".SZ"):
        return "china_a_shenzhen", "CNY"
    if s.endswith(".HK"):
        return "hong_kong", "HKD"
    if s.endswith(".T"):
        return "japan", "JPY"
    if s.endswith(".L"):
        return "uk", "GBP"
    if s.endswith(".TO"):
        return "canada", "CAD"
    if s.endswith("^") or s.startswith("^"):
        return "index", "USD"
    if s in {"SPY", "QQQ", "IWM", "DIA", "SOXQ", "VOO", "VTI"}:
        return "us_etf", "USD"
    if s:
        return "us_equity", "USD"
    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把 yfinance 单标的返回的 MultiIndex 列（如 ``('Close','MSFT')``）压平。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # 优先取 level 0（OHLCV 名称所在层）
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        rename[col] = _COLUMN_ALIASES.get(key, str(col).strip())
    return df.rename(columns=rename)


def _date_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """当没有日期列时，尝试从时间类型索引中取出日期。

    只有索引本身是时间类型才会这样做；否则明确报错，避免把普通数字索引
    （例如 ``RangeIndex``）误当成日期而产生 1970 年的假数据。
    """
    index = df.index
    is_datetime_index = isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(
        getattr(index, "dtype", None)
    )
    if not is_datetime_index:
        raise ValueError(
            f"行情数据缺少日期列（既没有 Date/Datetime 列，索引也不是时间类型）：现有列 {list(df.columns)}"
        )

    out = df.copy()
    out.index = pd.to_datetime(index)
    out.index.name = "Date"
    out = out.reset_index()
    out = _flatten_columns(out)
    return _canonicalize_columns(out)


def normalize_ohlcv(raw: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """把任意来源的原始行情规范化为标准 OHLCV DataFrame。

    处理内容：MultiIndex 列、日期列识别、列名别名、日期解析（含时区）、
    排序、去重。非法数值行的清洗由 :func:`drop_invalid_rows` 负责。

    返回的 DataFrame 带有 ``attrs['notes']``，记录规范化过程中的提示信息。
    """
    notes: List[str] = []
    if raw is None or len(raw) == 0:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        empty.attrs["notes"] = notes
        return empty

    df = _flatten_columns(raw)
    df = _canonicalize_columns(df)

    if "Date" not in df.columns:
        # 日期可能在时间类型索引上；若索引不是时间类型则明确报错
        df = _date_from_index(df)

    missing = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if "Date" in missing:
        raise ValueError(f"行情数据缺少日期列：现有列 {list(df.columns)}")
    if "Open" in missing or "Close" in missing:
        raise ValueError(f"行情数据缺少必要的价格列：缺少 {missing}")
    # High/Low 缺失时允许回退到 Close，但会记录提示（不同于 yfinance 的正常情况）
    for fallback in ("High", "Low"):
        if fallback in missing:
            df[fallback] = df["Close"]
            notes.append(f"原始数据缺少 {fallback} 列，已用 Close 回退填充")
    if "Volume" in missing:
        df["Volume"] = np.nan
        notes.append("原始数据缺少 Volume 列，已置为空值")

    # 日期解析
    date_raw = df["Date"]
    try:
        parsed = pd.to_datetime(date_raw, errors="coerce", utc=True)
    except (TypeError, ValueError):  # pragma: no cover - 极端输入兜底
        parsed = pd.to_datetime(date_raw, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        # 时区感知（通常出现在日内的数据）-> 去掉时区但保留所属交易日
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    bad_dates = int(parsed.isna().sum())
    if bad_dates:
        notes.append(f"丢弃 {bad_dates} 行无法解析的日期")
    df["Date"] = parsed
    df = df.loc[df["Date"].notna()]

    # 数值化
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[STANDARD_COLUMNS].copy()

    # 排序 + 去重（保留最后一条，视为修正后的数据）
    if not df["Date"].is_monotonic_increasing:
        df = df.sort_values("Date")
        notes.append("原始数据日期未按升序排列，已重新排序")
    dup_mask = df["Date"].duplicated(keep="last")
    dup_count = int(dup_mask.sum())
    if dup_count:
        notes.append(f"发现 {dup_count} 条重复日期记录，已保留最后一条")
        df = df.loc[~dup_mask]

    # 统一到“日期”（丢掉 00:00:00 时间部分）
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.reset_index(drop=True)
    df.attrs["notes"] = notes
    return df


def drop_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """丢弃明显非法的行情行（非正价格、High<Low、负成交量）。

    返回 ``(清洗后的 DataFrame, 提示信息列表)``。**不会**对缺失价格做任何填充。
    """
    notes: List[str] = []
    if df.empty:
        return df.copy(), notes

    out = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]

    # 完全缺失价格的空行
    all_nan = out[price_cols].isna().all(axis=1)
    n_all_nan = int(all_nan.sum())
    if n_all_nan:
        notes.append(f"丢弃 {n_all_nan} 行价格全部缺失的记录")
        out = out.loc[~all_nan]

    # 非正价格
    non_positive = (out[price_cols] <= 0).any(axis=1)
    n_non_positive = int(non_positive.sum())
    if n_non_positive:
        notes.append(f"丢弃 {n_non_positive} 行存在非正价格的记录（无效数据）")
        out = out.loc[~non_positive]

    # High < Low
    bad_range = out["High"] < out["Low"]
    n_bad_range = int((bad_range & out["High"].notna() & out["Low"].notna()).sum())
    if n_bad_range:
        notes.append(f"丢弃 {n_bad_range} 行 High < Low 的记录（无效数据）")
        out = out.loc[~(bad_range & out["High"].notna() & out["Low"].notna())]

    # 负成交量 -> 置为 NaN 并提示（不伪造数值）
    negative_volume = out["Volume"] < 0
    n_negative_volume = int(negative_volume.sum())
    if n_negative_volume:
        notes.append(f"{n_negative_volume} 行成交量为负，已置为空值")
        out.loc[negative_volume, "Volume"] = np.nan

    out = out.reset_index(drop=True)
    out.attrs["notes"] = list(df.attrs.get("notes", [])) + notes
    return out, notes


# ---------------------------------------------------------------------------
# 数据质量
# ---------------------------------------------------------------------------


@dataclass
class DataQuality:
    """行情数据质量报告。"""

    symbol: str
    bars: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    issues: List[str] = field(default_factory=list)  # 严重问题（可能导致分析不可靠）
    warnings: List[str] = field(default_factory=list)  # 非阻塞提示
    adjusted: bool = True
    adjustment_note: str = ADJUSTMENT_NOTE
    timezone: str = ""  # 交易所时区
    fetched_at_utc: str = ""
    # 数据源最新一根 K 线的日期（可能是盘中那根）
    source_last_bar_date: Optional[str] = None
    # 分析实际使用数据的最后一根 K 线日期
    last_bar_date: Optional[str] = None
    # 数据源最后一根 K 线是否属于“进行中的交易日”
    source_last_bar_incomplete: bool = False
    # 是否把该盘中 K 线排除出了指标计算
    intraday_excluded: bool = False
    # 分析所用数据的最后一根 K 线是否未完成（排除后为 False）
    incomplete_bar: bool = False
    incomplete_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "adjusted": self.adjusted,
            "adjustment_note": self.adjustment_note,
            "timezone": self.timezone,
            "fetched_at_utc": self.fetched_at_utc,
            "source_last_bar_date": self.source_last_bar_date,
            "last_bar_date": self.last_bar_date,
            "source_last_bar_incomplete": self.source_last_bar_incomplete,
            "intraday_excluded": self.intraday_excluded,
            "incomplete_bar": self.incomplete_bar,
            "incomplete_reason": self.incomplete_reason,
            "ok": self.ok,
        }

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        if self.issues:
            lines.extend(f"[问题] {msg}" for msg in self.issues)
        if self.warnings:
            lines.extend(f"[提示] {msg}" for msg in self.warnings)
        if not lines:
            lines.append("数据质量检查未发现异常。")
        return lines


def _reference_time(timezone_name: str, as_of: Optional[pd.Timestamp]) -> pd.Timestamp:
    """把参考时间规整为带时区的时间点（用于新鲜度与盘中判断）。

    * 未提供 ``as_of``：交易所当地时间的“现在”；
    * 只给了日期（如 ``--as-of 2026-10-09``）：视为该日**已收盘**（23:59），
      保证结果可复现，不会因为跑脚本的时刻不同而变化；
    * 给了具体时间：直接使用。
    """
    if as_of is None:
        return market_now(timezone_name)

    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        if ts.normalize() == ts:  # 只有日期，没有时间
            ts = ts + pd.Timedelta(hours=23, minutes=59)
        ts = ts.tz_localize(timezone_name)
    return ts


def _local_date(timezone_name: str, as_of: Optional[pd.Timestamp]) -> pd.Timestamp:
    """把参考时间换算成**交易所当地的日期**（naive，仅日期）。"""
    return _reference_time(timezone_name, as_of).tz_convert(timezone_name).normalize().tz_localize(None)


def check_quality(
    df: pd.DataFrame,
    symbol: str,
    min_bars: int = 260,
    timezone_name: str = "America/New_York",
    stale_days: int = 10,
    as_of: Optional[pd.Timestamp] = None,
    extra_notes: Optional[Sequence[str]] = None,
    close_time: str = "16:00",
    settle_minutes: int = DEFAULT_SETTLE_MINUTES,
) -> DataQuality:
    """对已清洗的行情做质量检查（只读，不修改数据）。

    参考时间（``as_of``）与“是否盘中”的判断全部基于**交易所当地时区**，
    而不是运行脚本的机器时区。
    """
    quality = DataQuality(symbol=symbol)
    quality.timezone = timezone_name
    quality.fetched_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for note in list(df.attrs.get("notes", [])) + list(extra_notes or []):
        if note not in quality.warnings:
            quality.warnings.append(note)

    if df is None or df.empty:
        quality.issues.append("未获取到任何行情数据")
        return quality

    quality.bars = int(len(df))
    quality.start_date = pd.Timestamp(df["Date"].iloc[0]).strftime("%Y-%m-%d")
    quality.end_date = pd.Timestamp(df["Date"].iloc[-1]).strftime("%Y-%m-%d")
    quality.source_last_bar_date = quality.end_date
    quality.last_bar_date = quality.end_date

    # 日期单调性 / 重复（normalize 已处理，这里作为最终断言）
    if not df["Date"].is_monotonic_increasing:
        quality.issues.append("日期未按升序排列")
    if df["Date"].duplicated().any():
        quality.issues.append("存在重复日期记录")

    # 缺失值
    null_counts = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum()
    for col, cnt in null_counts.items():
        cnt = int(cnt)
        if cnt == 0:
            continue
        if col in ("Open", "High", "Low", "Close"):
            quality.issues.append(f"{col} 存在 {cnt} 个缺失值（未做填充）")
        else:
            quality.warnings.append(f"{col} 存在 {cnt} 个缺失值（未做填充）")

    # 无效价格
    if (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).any():
        quality.issues.append("存在非正价格记录")
    if (df["High"] < df["Low"]).any():
        quality.issues.append("存在 High < Low 的记录")

    # 成交量
    if (df["Volume"].dropna() < 0).any():
        quality.issues.append("存在负成交量记录")
    zero_volume = int((df["Volume"].fillna(-1) == 0).sum())
    if zero_volume:
        quality.warnings.append(f"存在 {zero_volume} 个零成交量交易日（可能为停牌或数据缺失）")

    # 历史长度
    if quality.bars < min_bars:
        quality.issues.append(
            f"历史数据不足：仅有 {quality.bars} 根日 K，少于所需的 {min_bars} 根，"
            f"MA200 / 6 个月收益率等指标可能不可用"
        )

    # 数据新鲜度（相对交易所当地今天）
    reference = _reference_time(timezone_name, as_of)
    local_today = _local_date(timezone_name, as_of)
    last_date = pd.Timestamp(df["Date"].iloc[-1]).normalize()
    gap_days = int((local_today - last_date).days)
    if gap_days > stale_days:
        quality.warnings.append(
            f"最后一根 K 线为 {quality.end_date}，距交易所当地日期已 {gap_days} 天，数据可能不新鲜"
        )
    elif gap_days < 0:
        quality.warnings.append(
            f"最后一根 K 线（{quality.end_date}）晚于交易所当地日期（{local_today.strftime('%Y-%m-%d')}），请留意数据源"
        )

    # 最后一根 K 线是否属于“进行中的交易日”
    in_progress, reason = session_in_progress(
        last_date,
        timezone_name,
        close_time=close_time,
        settle_minutes=settle_minutes,
        now=reference,
    )
    quality.source_last_bar_incomplete = in_progress
    quality.incomplete_bar = in_progress  # 未排除前，分析数据本身就带着这根未完成的 K 线
    quality.incomplete_reason = reason
    if in_progress:
        quality.warnings.append(
            "最后一根 K 线（{}）属于进行中的交易日：{}。"
            "该 K 线的收盘价实为盘中最新价、成交量为盘中累计值，直接用于指标会失真。".format(
                quality.end_date, reason
            )
        )

    return quality


def split_incomplete_bar(
    daily: pd.DataFrame, quality: DataQuality
) -> Tuple[pd.DataFrame, Optional[Dict[str, object]]]:
    """若最后一根 K 线属于**进行中的交易日**，把它从分析数据中拆出来。

    返回 ``(用于计算的日线, 盘中快照或 None)``。

    盘中快照中的 ``last`` 是盘中最新价、``volume`` 是盘中累计成交量，二者都
    **不参与**任何指标计算；拆出后 ``quality`` 描述的是“实际用于分析的数据”。
    """
    if daily is None or daily.empty or not quality.source_last_bar_incomplete:
        return daily, None

    if len(daily) < 2:
        # 只剩这一根时无法拆分（没有上一根收盘价作对比），保持原样并保留警告
        return daily, None

    row = daily.iloc[-1]
    prev_close = float(daily["Close"].iloc[-2])
    last_price = float(row["Close"])
    volume = row.get("Volume")

    snapshot: Dict[str, object] = {
        "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
        "open": _safe_float(row.get("Open")),
        "high": _safe_float(row.get("High")),
        "low": _safe_float(row.get("Low")),
        "last": _safe_float(last_price),
        "volume": _safe_float(volume),
        "prev_close": _safe_float(prev_close),
        "change_pct_vs_prev_close": (
            None if prev_close in (0, None) else round((last_price / prev_close - 1.0) * 100.0, 2)
        ),
        "reason": quality.incomplete_reason,
        "note": "盘中数据：最新价为盘中现价，成交量为盘中累计值，未参与任何指标计算",
    }

    closed = daily.iloc[:-1].reset_index(drop=True)
    closed.attrs["notes"] = list(daily.attrs.get("notes", []))

    quality.intraday_excluded = True
    quality.incomplete_bar = False  # 分析实际使用的数据里已无未完成 K 线
    quality.bars = int(len(closed))
    quality.end_date = pd.Timestamp(closed["Date"].iloc[-1]).strftime("%Y-%m-%d")
    quality.last_bar_date = quality.end_date
    quality.warnings.append(
        "已把进行中的交易日 {}（盘中）排除出指标计算，分析数据截止 {}".format(
            snapshot["date"], quality.end_date
        )
    )
    return closed, snapshot


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, 4)


# ---------------------------------------------------------------------------
# 获取
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """调用 yfinance 获取原始行情（不做清洗）。

    指定 ``start`` / ``end``（``YYYY-MM-DD``）时按日期区间取数（用于"按历史日期分析"），
    否则按 ``period`` 取数。
    """
    import yfinance as yf  # 延迟导入，便于在不联网的环境下复用本模块的纯计算函数

    logger.debug(
        "抓取行情 %s period=%s interval=%s start=%s end=%s", symbol, period, interval, start, end
    )
    kwargs: Dict[str, object] = {
        "interval": interval,
        "auto_adjust": auto_adjust,
        "actions": False,
        "progress": False,
        "threads": False,
    }
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    raw = yf.download(symbol, **kwargs)
    return raw


def read_csv_file(path: str) -> pd.DataFrame:
    """从本地 CSV 读取行情（用于离线分析 / 复用已有文件）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"行情文件不存在：{path}")
    return pd.read_csv(path)


def prepare_daily(
    raw: pd.DataFrame,
    symbol: str,
    min_bars: int = 260,
    timezone_name: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
    close_time: str = "16:00",
) -> Tuple[pd.DataFrame, DataQuality]:
    """规范化 + 清洗 + 质量检查，返回 ``(daily_df, quality)``。

    ``timezone_name`` 缺省时根据代码自动推断交易所时区（见 :func:`infer_timezone`）。
    """
    timezone_name = timezone_name or infer_timezone(symbol)
    df = normalize_ohlcv(raw, symbol)
    df, clean_notes = drop_invalid_rows(df)
    quality = check_quality(
        df,
        symbol,
        min_bars=min_bars,
        timezone_name=timezone_name,
        as_of=as_of,
        extra_notes=clean_notes,
        close_time=close_time,
    )
    return df, quality


def fetch_daily(
    symbol: str,
    period: str = "2y",
    min_bars: int = 260,
    timezone_name: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, DataQuality]:
    """抓取 + 规范化的日线数据（自动复权）。"""
    raw = fetch_ohlcv(symbol, period=period, interval="1d", auto_adjust=True, start=start, end=end)
    return prepare_daily(
        raw,
        symbol,
        min_bars=min_bars,
        timezone_name=timezone_name,
        as_of=as_of,
    )


# ---------------------------------------------------------------------------
# 日线 -> 周线聚合
# ---------------------------------------------------------------------------


def _iso_key(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = pd.Timestamp(ts).isocalendar()
    return int(iso[0]), int(iso[1])


def _aggregate_one_week(group: pd.DataFrame) -> Dict[str, object]:
    """按周聚合的单周计算逻辑。

    规则（与需求一致）：

    * Open  = 该周第一根**有效**日 K 线的开盘价
    * High  = 该周所有日 K 线最高价的最大值
    * Low   = 该周所有日 K 线最低价的最小值
    * Close = 该周最后一根**有效**日 K 线的收盘价
    * Volume= 该周成交量之和
    """
    g = group.sort_values("Date")
    dates = g["Date"]
    opens = g["Open"].dropna()
    closes = g["Close"].dropna()
    highs = g["High"].dropna()
    lows = g["Low"].dropna()
    volumes = g["Volume"].dropna()

    return {
        "Date": pd.Timestamp(dates.max()),
        "WeekStart": pd.Timestamp(dates.min()),
        "WeekEnd": pd.Timestamp(dates.max()),
        "Open": float(opens.iloc[0]) if len(opens) else np.nan,
        "High": float(highs.max()) if len(highs) else np.nan,
        "Low": float(lows.min()) if len(lows) else np.nan,
        "Close": float(closes.iloc[-1]) if len(closes) else np.nan,
        "Volume": float(volumes.sum()) if len(volumes) else np.nan,
        "Bars": int(len(g)),
    }


def is_week_complete(week_end: pd.Timestamp, as_of: Optional[pd.Timestamp] = None) -> bool:
    """判断某一周是否已经结束（不是“进行中的交易周”）。

    规则：

    1. 若该周最后一根日 K 落在周五，视为已完成（正常交易周）。
    2. 若参考日期 ``as_of`` 落在**更晚**的 ISO 周，则该周已经过去，视为已完成
       （覆盖“周五为节假日，本周提前收市”的情况）。
    3. 其余情况（尤其是最后一根日 K 就是当天）视为未结束。
    """
    week_end = pd.Timestamp(week_end).normalize()
    if week_end.weekday() == 4:  # 周五
        return True
    if as_of is not None and _iso_key(pd.Timestamp(as_of).normalize()) > _iso_key(week_end):
        return True
    return False


def aggregate_weekly(
    daily: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    drop_incomplete: bool = True,
    timezone_name: str = "America/New_York",
) -> Tuple[pd.DataFrame, bool, List[str]]:
    """把日线聚合为周线。

    返回 ``(weekly_df, last_week_complete, warnings)``。

    * ``weekly_df`` 的 ``Date`` 列 = 该周最后一个交易日的日期。
    * 当 ``drop_incomplete=True``（默认）时，尚未结束的交易周会从结果中移除，
      避免把“进行中的周”当成已完成周线用于趋势判断。
    """
    warnings: List[str] = []
    columns = [
        "Date",
        "WeekStart",
        "WeekEnd",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Bars",
    ]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=columns), False, ["日线数据为空，无法聚合周线"]

    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    iso = df["Date"].dt.isocalendar()
    df["_iso_year"] = iso["year"].astype(int).to_numpy()
    df["_iso_week"] = iso["week"].astype(int).to_numpy()

    rows = [
        _aggregate_one_week(group)
        for _, group in df.groupby(["_iso_year", "_iso_week"], sort=True)
    ]
    weekly = pd.DataFrame(rows, columns=columns)
    if weekly.empty:
        return weekly, False, ["周线聚合结果为空"]

    last_week_end = pd.Timestamp(weekly["WeekEnd"].iloc[-1])
    reference = _local_date(timezone_name, as_of)
    last_week_complete = is_week_complete(last_week_end, reference)

    if weekly["Bars"].iloc[-1] < 5:
        warnings.append(
            f"最后一个交易周仅有 {int(weekly['Bars'].iloc[-1])} 个交易日"
            f"（WeekEnd={last_week_end.strftime('%Y-%m-%d')}）"
            + ("" if last_week_complete else "，属于进行中的交易周")
        )

    if not last_week_complete:
        warnings.append(
            f"最近一周（截至 {last_week_end.strftime('%Y-%m-%d')}）尚未结束，"
            "已从周线指标计算中排除，避免使用未来数据"
        )
        if drop_incomplete:
            weekly = weekly.iloc[:-1].reset_index(drop=True)

    weekly["Date"] = pd.to_datetime(weekly["Date"]).dt.normalize()
    weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"]).dt.normalize()
    weekly["WeekEnd"] = pd.to_datetime(weekly["WeekEnd"]).dt.normalize()
    weekly = weekly.reset_index(drop=True)
    weekly.attrs["warnings"] = warnings
    weekly.attrs["last_week_complete"] = last_week_complete
    return weekly, last_week_complete, warnings


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------


def to_csv_frame(df: pd.DataFrame, price_decimals: int = 2) -> pd.DataFrame:
    """转换成用于 CSV 输出的 DataFrame（Date 为 ``YYYY-MM-DD`` 字符串）。

    与原有 ``fetch_data.py`` 的输出格式保持一致。
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col].round(price_decimals)
    return out
