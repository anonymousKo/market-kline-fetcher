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
    timezone: str = ""
    fetched_at_utc: str = ""
    last_bar_date: Optional[str] = None
    incomplete_bar: bool = False

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
            "last_bar_date": self.last_bar_date,
            "incomplete_bar": self.incomplete_bar,
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


def check_quality(
    df: pd.DataFrame,
    symbol: str,
    min_bars: int = 260,
    timezone_name: str = "America/New_York",
    stale_days: int = 10,
    as_of: Optional[pd.Timestamp] = None,
    extra_notes: Optional[Sequence[str]] = None,
) -> DataQuality:
    """对已清洗的行情做质量检查（只读，不修改数据）。"""
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

    # 数据新鲜度
    reference = as_of if as_of is not None else pd.Timestamp.now().normalize()
    last_date = pd.Timestamp(df["Date"].iloc[-1]).normalize()
    gap_days = int((reference.normalize() - last_date).days)
    if gap_days > stale_days:
        quality.warnings.append(
            f"最后一根 K 线为 {quality.end_date}，距参考日期已 {gap_days} 天，数据可能不新鲜"
        )

    # 当日是否可能尚未收盘
    if gap_days == 0:
        quality.incomplete_bar = True
        quality.warnings.append(
            "最后一根 K 线日期为参考日期当天，该交易日可能尚未收盘（当日数据可能不完整）"
        )

    return quality


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
    timezone_name: str = "America/New_York",
    as_of: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, DataQuality]:
    """规范化 + 清洗 + 质量检查，返回 ``(daily_df, quality)``。"""
    df = normalize_ohlcv(raw, symbol)
    df, clean_notes = drop_invalid_rows(df)
    quality = check_quality(
        df,
        symbol,
        min_bars=min_bars,
        timezone_name=timezone_name,
        as_of=as_of,
        extra_notes=clean_notes,
    )
    return df, quality


def fetch_daily(
    symbol: str,
    period: str = "2y",
    min_bars: int = 260,
    timezone_name: str = "America/New_York",
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
    reference = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now().normalize()
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
