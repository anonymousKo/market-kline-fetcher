"""配置管理。

设计原则
--------
1. 任何密钥都只从环境变量（或 ``.env`` 文件）读取，绝不硬编码在源码中。
2. 缺少密钥时程序依然可以运行行情获取与指标计算，只是跳过 AI 分析。
3. 模型 / 服务商完全可配置（任何兼容 OpenAI Chat Completions 协议的服务）。

所有配置项见项目根目录的 ``.env.example``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:  # python-dotenv 是可选的，缺失时退化为只读环境变量
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 依赖缺失时的兜底
    load_dotenv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 环境变量读取工具
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    items = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return items or list(default)


def load_env(dotenv_path: Optional[str] = None) -> None:
    """加载 ``.env``（如果存在）。

    优先级：真实环境变量 > ``.env`` 文件。``load_dotenv`` 默认不覆盖已存在的
    环境变量，因此 CI（GitHub Actions Secrets）中的变量始终优先生效。
    """
    if load_dotenv is None:
        return
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    # 依次尝试当前工作目录与项目根目录
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (os.path.join(os.getcwd(), ".env"), os.path.join(root, ".env")):
        if os.path.isfile(candidate):
            load_dotenv(dotenv_path=candidate, override=False)


# ---------------------------------------------------------------------------
# 配置对象
# ---------------------------------------------------------------------------


@dataclass
class AISettings:
    """大模型相关配置。"""

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    # 单次请求超时（秒）
    timeout: float = 60.0
    # 失败后的额外重试次数（0 表示只尝试一次）。仅对超时 / 限流 / 网络错误重试。
    max_retries: int = 2
    temperature: float = 0.2
    max_tokens: int = 8000
    # 是否请求服务端返回严格 JSON（部分兼容服务不支持，失败会自动降级）
    json_mode: bool = True
    # 重试基础退避时间（秒），实际等待为 base * 2**attempt
    retry_backoff: float = 2.0

    @property
    def configured(self) -> bool:
        """是否具备调用大模型的最小条件。"""
        return bool(self.api_key)

    def masked_key(self) -> str:
        """返回脱敏后的密钥，用于日志。"""
        return mask_secret(self.api_key)


@dataclass
class Settings:
    """全局配置。"""

    ai: AISettings = field(default_factory=AISettings)
    # 分析用日线数据抓取范围（yfinance period 语法）
    analysis_period: str = "2y"
    # 计算 MA200 / 6 个月收益率所需的最少日线数量
    min_daily_bars: int = 260
    # 相对强弱对比基准
    benchmarks: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    # 输出目录
    kline_dir: str = "kline_data"
    reports_dir: str = "reports"
    # 数据时区（仅用于记录与展示；日线数据本身按交易所本地日期存储）
    timezone: str = "America/New_York"

    @classmethod
    def from_env(cls) -> "Settings":
        ai = AISettings(
            api_key=_get_str("AI_API_KEY") or _get_str("OPENAI_API_KEY"),
            base_url=_get_str("AI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            model=_get_str("AI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            timeout=_get_float("AI_TIMEOUT", 60.0),
            max_retries=_get_int("AI_MAX_RETRIES", 2),
            temperature=_get_float("AI_TEMPERATURE", 0.2),
            max_tokens=_get_int("AI_MAX_TOKENS", 8000),
            json_mode=_get_bool("AI_JSON_MODE", True),
            retry_backoff=_get_float("AI_RETRY_BACKOFF", 2.0),
        )
        return cls(
            ai=ai,
            analysis_period=_get_str("ANALYSIS_PERIOD", "2y") or "2y",
            min_daily_bars=_get_int("MIN_DAILY_BARS", 260),
            benchmarks=_get_list("BENCHMARKS", ["SPY", "QQQ"]),
            kline_dir=_get_str("KLINE_DIR", "kline_data") or "kline_data",
            reports_dir=_get_str("REPORTS_DIR", "reports") or "reports",
            timezone=_get_str("MARKET_TIMEZONE", "America/New_York") or "America/New_York",
        )


def mask_secret(value: Optional[str]) -> str:
    """把密钥脱敏成 ``sk-a***xyz`` 形式，避免写入日志。"""
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "*" * len(value)
    return "{}***{}".format(value[:4], value[-4:])
