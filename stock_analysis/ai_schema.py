"""大模型返回结果的解析、校验与归一化。

设计目标
--------
* 字段名稳定：下游（报告渲染、历史记录）只依赖归一化后的固定结构。
* 校验必填字段；缺失字段用**明确的空值**补齐，并记录 warning。
* 处理非法 JSON（含 Markdown 代码块包裹、前后多余文字等常见情况）。
* 绝不把未经验证的模型输出直接当作可靠数据：所有数值型字段都做类型检查，
  无法解析时置为 None 而不是猜测。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# 归一化后的固定结构（下游依赖这些字段名）
_SCENARIO_SCHEMA: Dict[str, Any] = {
    "conditions": "list",
    "watch": "list",
    "invalidation": "list",
}

SCHEMA: Dict[str, Any] = {
    "trend": {
        "weekly_trend": "str",
        "daily_trend": "str",
        "aligned": "bool",
        "alignment_comment": "str",
        "evidence": "list",
    },
    "momentum": {
        "rsi_state": "str",
        "macd_state": "str",
        "momentum_direction": "str",
        "divergence": {"has_evidence": "bool", "comment": "str"},
    },
    "volume": {
        "volume_relation": "str",
        "price_volume_support": "str",
        "summary": "str",
    },
    "relative_strength": {
        "benchmark": "str",
        "period": "str",
        "outperforming": "str",
        "summary": "str",
    },
    "key_levels": {
        "support": "level_list",
        "resistance": "level_list",
        "notes": "str",
    },
    "scenarios": {
        "bullish": _SCENARIO_SCHEMA,
        "bearish": _SCENARIO_SCHEMA,
        "neutral": _SCENARIO_SCHEMA,
    },
    "summary": {
        "technical_state": "str",
        "top_signals": "list",
        "main_risks": "list",
        "watch_conditions": "list",
    },
    "disclaimer": "str",
}

# 这些顶层字段缺失时视为"模型返回不合格"，需要报错而不是静默降级
REQUIRED_CORE_KEYS = ("trend", "momentum", "volume", "scenarios", "summary")

_UNKNOWN = "未提供"


class AIResultError(RuntimeError):
    """模型返回内容无法解析为符合约定的 JSON。

    异常上附带：

    * ``raw_response``：模型的**原始返回文本**（便于排查与落盘）；
    * ``meta``：调用元信息（如 ``finish_reason`` / token 数），可能为空字典。
    """

    def __init__(
        self,
        message: str,
        raw_response: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.meta: Dict[str, Any] = dict(meta or {})


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z]*[ \t]*\r?\n?")

# 扫描“夹带在说明文字里的 JSON”时，候选对象至少要命中这么多个核心字段，
# 否则很容易把某个子对象（例如 relative_strength 里也有 summary 键）误当成结果
MIN_SCANNED_CORE_HITS = 2

# JSON Schema 定义中常见的标记字段（用于识别“模型把 schema 当结果返回”的情况）
_SCHEMA_MARKER_KEYS = {"type", "properties", "required", "$schema", "additionalproperties", "items"}

_MAX_RAW_CHARS = 20000


def _truncate(text: Optional[str], limit: int = _MAX_RAW_CHARS) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(已截断，共 {len(text)} 字符)"


def preview_text(text: Any, limit: int = 300) -> str:
    """把任意内容压成一行短预览，用于日志与错误信息。"""
    flat = str(text).replace("\r", " ").replace("\n", " ").strip()
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _try_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _NOTHING


class _Nothing:
    """``json.loads`` 失败的哨兵（区别于合法的 ``None``）。"""


_NOTHING = _Nothing()


def _iter_json_objects(text: str, max_objects: int = 50) -> List[Dict[str, Any]]:
    """扫描文本中**所有**可独立解析的 JSON 对象（按出现顺序）。

    这样即使模型在 JSON 前后写了说明文字、且说明里也包含了花括号，
    也不会把第一个碰到的“假对象”当成结果（见 :func:`_select_best_object`）。
    """
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    index = text.find("{")
    while index != -1 and len(found) < max_objects:
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(parsed, dict):
            found.append(parsed)
            index = text.find("{", index + max(int(end), 1))
        else:
            index = text.find("{", index + 1)
    return found


def _core_hits(obj: Dict[str, Any]) -> int:
    return sum(1 for key in REQUIRED_CORE_KEYS if key in obj)


def _select_best_object(
    candidates: List[Dict[str, Any]], min_core_hits: int = 0
) -> Optional[Dict[str, Any]]:
    """从多个候选对象中挑出最像“分析结果”的那个。

    优先选择包含最多核心字段的对象；相同时取内容最长的（避免选中说明文字里的
    小对象）。``min_core_hits`` 用于排除明显不合格的子对象。
    """
    usable = [obj for obj in candidates if _core_hits(obj) >= min_core_hits]
    if not usable:
        return None

    def sort_key(obj: Dict[str, Any]):
        return (
            _core_hits(obj),
            len(json.dumps(obj, ensure_ascii=False, default=str)),
        )

    return max(usable, key=sort_key)


def looks_like_json_schema(obj: Any) -> bool:
    """判断对象是否为 JSON Schema 定义片段，而不是分析数据。"""
    if not isinstance(obj, dict):
        return False
    keys = {str(key).strip().lower() for key in obj}
    return bool(keys & _SCHEMA_MARKER_KEYS)


def _contains_schema_sections(obj: Dict[str, Any]) -> bool:
    """判断“看似分析结果”的对象是否其实由 schema 片段组成。"""
    schema_like = sum(1 for value in obj.values() if looks_like_json_schema(value))
    return schema_like >= max(1, len(obj) // 2)


def _decode_prefix(text: str) -> Tuple[Any, Optional[str]]:
    """尝试在文本开头解析一个 JSON 值，返回 ``(对象, 错误描述)``。"""
    try:
        obj, _end = json.JSONDecoder().raw_decode(text.lstrip())
        return obj, None
    except json.JSONDecodeError as exc:
        return _NOTHING, "{}（第 {} 行第 {} 列，字符位置 {}）".format(
            exc.msg, exc.lineno, exc.colno, exc.pos
        )


def _strip_fence(text: str) -> str:
    """去掉可能存在的代码块包裹（包括未闭合的情况）。"""
    body = _FENCE_OPEN_RE.sub("", text, count=1).rstrip()
    if body.endswith("```"):
        body = body[:-3].rstrip()
    return body


def _malformed_json_error(raw: str, body: str, detail: Optional[str]) -> AIResultError:
    return AIResultError(
        "模型返回的 JSON 不完整或格式错误：{}。已读入 {} 个字符。"
        "最可能的原因是输出达到 token 上限被截断，请调大 AI_MAX_TOKENS 后重试；"
        "完整原始返回见记录中的 ai_raw_response 字段。".format(
            detail or "无法解析", len(body)
        ),
        raw,
    )


def parse_json_response(text: Any) -> Dict[str, Any]:
    """把模型返回的文本解析为 dict。

    解析优先级：

    1. 整段文本就是一个 JSON 对象；
    2. 文本以 JSON 对象开头（后面可能跟了多余文字）；
    3. Markdown 代码块里的 JSON 对象；
    4. 说明文字中夹带的 JSON 对象（要求至少命中两个核心字段，避免误选子对象）。

    如果文本本身就是一段以 ``{`` 开头的 JSON 但无法解析（典型原因是输出被截断），
    **不会**退而选择其中的子对象，而是明确报“JSON 不完整”。
    """
    if isinstance(text, dict):
        return text
    if text is None:
        raise AIResultError("模型返回内容为空")

    raw = str(text).strip()
    if not raw:
        raise AIResultError("模型返回内容为空", str(text))

    # 1) 整段解析
    whole = _try_load_json(raw)
    if isinstance(whole, dict):
        return whole
    if whole is not _NOTHING and whole is not None:
        raise AIResultError(
            "模型返回的 JSON 顶层不是对象，而是 {}。原始内容开头：{}".format(
                type(whole).__name__, preview_text(raw)
            ),
            raw,
        )

    # 2) 去掉代码块后，正文本身是否以 { 开头（截断检测的关键分支）
    body = _strip_fence(raw)
    if body.lstrip().startswith("{"):
        obj, detail = _decode_prefix(body)
        if isinstance(obj, dict):
            return obj  # JSON 完整，只是后面多了说明文字
        raise _malformed_json_error(raw, body, detail)

    # 3) 代码块
    for match in _FENCE_RE.finditer(raw):
        fenced = _try_load_json(match.group(1).strip())
        if isinstance(fenced, dict):
            return fenced

    # 4) 说明文字中夹带的 JSON（要求至少命中两个核心字段，避免误选子对象）
    candidates = _iter_json_objects(raw)
    best = _select_best_object(candidates, min_core_hits=MIN_SCANNED_CORE_HITS)
    if best is None:
        raise AIResultError(
            "无法从模型返回内容中解析出符合预期的分析 JSON（共扫描到 {} 个 JSON 对象，"
            "但均未包含足够的必需字段）。原始内容开头：{}".format(len(candidates), preview_text(raw)),
            raw,
        )
    return best


# ---------------------------------------------------------------------------
# 归一化与校验
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1", "是"}:
            return True
        if token in {"false", "no", "n", "0", "否"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalize_level_list(value: Any, path: str, warnings: List[str]) -> List[Dict[str, Any]]:
    items = _as_list(value)
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            level = _as_float(item.get("level"))
            basis = _as_str(item.get("basis"))
            source = _as_str(item.get("source")) or ("程序计算" if level is not None else _UNKNOWN)
            if level is None and not basis:
                warnings.append(f"{path}[{index}] 既无 level 也无 basis，已跳过")
                continue
            normalized.append({"level": level, "basis": basis, "source": source})
        elif isinstance(item, (int, float)):
            normalized.append({"level": float(item), "basis": "", "source": _UNKNOWN})
        else:
            text = _as_str(item)
            if not text:
                continue
            warnings.append(f"{path}[{index}] 不是对象，已转换为文本说明")
            normalized.append({"level": None, "basis": text, "source": _UNKNOWN})
    return normalized


def _normalize_node(value: Any, schema: Any, path: str, warnings: List[str]) -> Any:
    if isinstance(schema, dict):
        node = value if isinstance(value, dict) else None
        if node is None:
            if value is not None:
                warnings.append(f"{path} 期望对象，实际为 {type(value).__name__}，已用空对象替代")
            else:
                warnings.append(f"{path} 缺失，已用空对象替代")
            node = {}
        result: Dict[str, Any] = {}
        for key, sub_schema in schema.items():
            child_path = f"{path}.{key}"
            if key not in node:
                warnings.append(f"缺少字段 {child_path}，已用默认值替代")
                result[key] = _normalize_node(None, sub_schema, child_path, [])
                continue
            result[key] = _normalize_node(node[key], sub_schema, child_path, warnings)
        return result

    if schema == "str":
        if value is None:
            return ""
        if not isinstance(value, str):
            warnings.append(f"{path} 期望字符串，实际为 {type(value).__name__}，已转换为文本")
        return _as_str(value)

    if schema == "bool":
        if value is None:
            return False
        parsed = _as_bool(value)
        if parsed is None:
            warnings.append(f"{path} 期望布尔值，实际为 {value!r}，已置为 false")
            return False
        return parsed

    if schema == "list":
        if value is None:
            return []
        if not isinstance(value, list):
            warnings.append(f"{path} 期望数组，实际为 {type(value).__name__}，已自动包装为数组")
        return [_as_str(item) for item in _as_list(value) if _as_str(item)]

    if schema == "level_list":
        return _normalize_level_list(value, path, warnings)

    raise ValueError(f"未知的 schema 类型：{schema}")  # pragma: no cover


def _find_nested_analysis(
    obj: Dict[str, Any], max_depth: int = 3
) -> Tuple[Optional[Dict[str, Any]], str]:
    """在包裹层中寻找真正的分析结果，例如 ``{"analysis": {"trend": ...}}`` 里的内层对象。"""
    queue: List[Tuple[Dict[str, Any], str, int]] = [(obj, "$", 0)]
    while queue:
        node, path, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for key, value in node.items():
            if not isinstance(value, dict):
                continue
            child_path = f"{path}.{key}"
            if (
                _core_hits(value) > 0
                and not looks_like_json_schema(value)
                and not _contains_schema_sections(value)
            ):
                return value, child_path
            queue.append((value, child_path, depth + 1))
    return None, ""


def _resolve_envelope(raw: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    """处理两种常见的异常结构：模型回显 JSON Schema、模型把结果多包了一层。"""
    if _core_hits(raw) > 0:
        return raw

    if looks_like_json_schema(raw):
        raise AIResultError(
            "模型返回的是 JSON Schema 定义而不是分析结果（实际顶层字段：{}）。"
            "这通常意味着模型未严格遵循指令。建议更换指令遵循能力更强的模型，"
            "或降低 AI_TEMPERATURE 后重试。".format(", ".join(map(str, list(raw.keys())[:10])))
        )

    inner, path = _find_nested_analysis(raw)
    if inner is not None:
        warnings.append(f"模型返回结果多包了一层 {path}，已自动解包")
        return inner

    return raw


def _missing_core_error(raw: Dict[str, Any]) -> AIResultError:
    missing = [key for key in REQUIRED_CORE_KEYS if key not in raw]
    actual = ", ".join(map(str, list(raw.keys())[:12])) or "（无）"
    dumped = json.dumps(raw, ensure_ascii=False, default=str)
    return AIResultError(
        "模型返回结果缺少必需的顶层字段：{}。实际返回的顶层字段为：[{}]。"
        "常见原因：模型未按要求使用英文字段名（例如改用了中文键名）、把结果包在外层对象中、"
        "或只返回了部分内容。原始返回开头：{}".format(
            ", ".join(missing), actual, preview_text(dumped, 300)
        )
    )


def normalize_result(raw: Any, strict: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """校验并归一化模型返回结果。

    返回 ``(归一化结果, warnings)``；当核心字段整体缺失时抛出
    :class:`AIResultError`（``strict=True`` 时任何字段缺失都会抛错）。
    """
    warnings: List[str] = []
    if not isinstance(raw, dict):
        raise AIResultError(f"模型返回结果不是 JSON 对象，而是 {type(raw).__name__}")

    raw = _resolve_envelope(raw, warnings)

    missing_core = [key for key in REQUIRED_CORE_KEYS if key not in raw]
    if missing_core:
        raise _missing_core_error(raw)

    extra_keys = [key for key in raw.keys() if key not in SCHEMA]
    if extra_keys:
        warnings.append(f"模型返回结果包含未定义的字段，已忽略：{', '.join(map(str, extra_keys))}")

    if strict:
        for key, sub_schema in SCHEMA.items():
            if key not in raw:
                raise AIResultError(f"缺少字段 {key}（strict 模式）")

    normalized = _normalize_node(raw, SCHEMA, "$", warnings)

    if strict and warnings:
        raise AIResultError("返回结果格式不符合约定：" + "；".join(warnings[:5]))

    return normalized, warnings


def validate_and_normalize(text: Any, strict: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """解析 + 校验 + 归一化的组合入口。

    失败时会把模型的**原始返回文本**附加到异常上（``AIResultError.raw_response``），
    便于日志输出与落盘排查。
    """
    raw_text = text if isinstance(text, str) else None
    try:
        parsed = parse_json_response(text)
        return normalize_result(parsed, strict=strict)
    except AIResultError as exc:
        if exc.raw_response is None and raw_text is not None:
            exc.raw_response = raw_text
        raise
