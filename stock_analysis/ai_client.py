"""大模型 API 客户端（兼容 OpenAI Chat Completions 协议）。

要点
----
* **不在源码中硬编码任何密钥**，只从 :class:`~stock_analysis.config.AISettings` 读取
  （其来源是环境变量或 ``.env``）。
* 服务商与模型完全可配置（OpenAI / DeepSeek / Moonshot / 本地 vLLM 等均可）。
* 设置明确的超时；对限流、网络异常、鉴权失败分别给出清晰的中文提示。
* **不做无限重试**：只对限流 / 超时 / 网络 / 5xx 这类瞬时错误做有限次（默认 2 次）重试，
  且采用指数退避；鉴权失败、请求非法等错误立即抛出。
* 支持注入 ``completion_fn``，便于单元测试（测试中绝不真实消耗 API 额度）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ai_schema import AIResultError, preview_text, validate_and_normalize
from .config import AISettings, mask_secret

logger = logging.getLogger(__name__)

# completion_fn(messages, **kwargs) -> 原始响应对象或字符串
CompletionFn = Callable[..., Any]


class AIError(RuntimeError):
    """AI 调用相关的基类异常。"""

    retryable = False


class AIConfigError(AIError):
    """配置不完整（例如缺少 API Key）。"""


class AIAuthError(AIError):
    """鉴权失败（API Key 无效或权限不足）。"""


class AIRateLimitError(AIError):
    """触发限流。"""

    retryable = True


class AITimeoutError(AIError):
    """请求超时。"""

    retryable = True


class AIConnectionError(AIError):
    """网络连接异常。"""

    retryable = True


class AIServerError(AIError):
    """服务端 5xx 错误。"""

    retryable = True


class AIBadRequestError(AIError):
    """请求非法（例如模型不存在、参数不被支持）。"""


class LLMClient:
    """极简的 Chat Completions 客户端。

    参数
    ----
    settings:
        :class:`~stock_analysis.config.AISettings`。
    completion_fn:
        可选注入。签名 ``fn(**kwargs) -> response``，其中 ``kwargs`` 包含
        ``model`` / ``messages`` / ``temperature`` / ``max_tokens`` /
        ``response_format`` / ``timeout``。用于测试时替换真实网络调用。
    sleep_fn:
        可选注入的 sleep 函数，用于测试退避逻辑而不真正等待。
    """

    def __init__(
        self,
        settings: AISettings,
        completion_fn: Optional[CompletionFn] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._completion_fn = completion_fn
        self._sleep = sleep_fn
        self._client: Any = None
        self._json_mode_active = bool(settings.json_mode)

    # ------------------------------------------------------------------
    # 配置检查
    # ------------------------------------------------------------------

    def ensure_configured(self) -> None:
        """检查是否具备调用条件，缺失时抛出带指引的异常。"""
        if not self.settings.configured:
            raise AIConfigError(
                "未配置大模型 API 密钥，已跳过 AI 分析。"
                "请设置环境变量 AI_API_KEY（或 OPENAI_API_KEY），"
                "可参考项目根目录的 .env.example。"
            )
        if not self.settings.model:
            raise AIConfigError("未配置模型名称，请设置环境变量 AI_MODEL。")

    # ------------------------------------------------------------------
    # 底层调用
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise AIConfigError(
                "未安装 openai 依赖，无法调用大模型 API。请执行 pip install -r requirements.txt"
            ) from exc

        logger.info(
            "初始化大模型客户端：model=%s base_url=%s api_key=%s",
            self.settings.model,
            self.settings.base_url,
            self.settings.masked_key(),
        )
        # max_retries=0：重试逻辑由本类统一控制，避免 SDK 与本类叠加导致次数不可控
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            max_retries=0,
        )
        return self._client

    def _build_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "timeout": self.settings.timeout,
        }
        if self.settings.max_tokens:
            kwargs["max_tokens"] = self.settings.max_tokens
        if self._json_mode_active:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _invoke(self, messages: List[Dict[str, str]]) -> Any:
        """执行一次底层调用（真实或注入的 mock）。"""
        kwargs = self._build_kwargs(messages)
        if self._completion_fn is not None:
            try:
                return self._completion_fn(**kwargs)
            except AIError:
                raise
            except Exception as exc:  # 注入函数抛出的异常也做统一映射
                raise self._map_exception(exc) from exc

        client = self._ensure_client()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._map_exception(exc) from exc

    @staticmethod
    def _map_exception(exc: Exception) -> AIError:
        """把 SDK 异常映射为带清晰中文提示的本项目异常。"""
        try:
            import openai
        except ImportError:  # pragma: no cover
            return AIError(f"AI 调用失败：{exc}")

        if isinstance(exc, openai.AuthenticationError):
            return AIAuthError("AI 鉴权失败：API Key 无效或权限不足，请检查 AI_API_KEY 配置。")
        if isinstance(exc, openai.RateLimitError):
            return AIRateLimitError("AI 接口触发限流（429），请稍后重试或降低调用频率。")
        if isinstance(exc, openai.APITimeoutError):
            return AITimeoutError(
                f"AI 请求超时（当前 AI_TIMEOUT={getattr(exc, 'timeout', None) or '未知'} 秒），"
                "可考虑调大 AI_TIMEOUT。"
            )
        if isinstance(exc, openai.APIConnectionError):
            return AIConnectionError(f"AI 接口网络连接失败：{exc}")
        if isinstance(exc, openai.BadRequestError):
            return AIBadRequestError(f"AI 请求被拒绝（可能是模型名或参数不受支持）：{exc}")
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and status >= 500:
                return AIServerError(f"AI 服务端错误（HTTP {status}）：{exc}")
            return AIBadRequestError(f"AI 接口返回错误（HTTP {status}）：{exc}")
        if isinstance(exc, openai.APIError):
            return AIError(f"AI 接口调用失败：{exc}")
        if isinstance(exc, (TimeoutError,)):
            return AITimeoutError("AI 请求超时。")
        return AIError(f"AI 调用出现未预期的错误：{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_content: str) -> Tuple[str, Dict[str, Any]]:
        """发起一次对话补全，返回 ``(文本内容, 元信息)``。"""
        self.ensure_configured()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        attempts = max(1, int(self.settings.max_retries) + 1)
        last_error: Optional[AIError] = None
        started = time.monotonic()

        for attempt in range(1, attempts + 1):
            try:
                response = self._invoke_with_json_fallback(messages)
                text = self._extract_text(response)
                return text, self._build_meta(response, attempt, time.monotonic() - started)
            except AIError as exc:
                last_error = exc
                if not exc.retryable or attempt >= attempts:
                    raise
                wait = max(0.0, self.settings.retry_backoff) * (2 ** (attempt - 1))
                logger.warning(
                    "AI 调用失败（第 %d/%d 次）：%s；%.1f 秒后重试",
                    attempt,
                    attempts,
                    exc,
                    wait,
                )
                self._sleep(wait)

        raise last_error or AIError("AI 调用失败")  # pragma: no cover

    def _invoke_with_json_fallback(self, messages: List[Dict[str, str]]) -> Any:
        try:
            return self._invoke(messages)
        except AIBadRequestError:
            if self._json_mode_active:
                logger.warning("服务端不支持 JSON 响应模式，降级为普通文本模式后重试一次")
                self._json_mode_active = False
                return self._invoke(messages)
            raise

    def complete_json(
        self,
        system_prompt: str,
        user_content: str,
        strict: bool = False,
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        """调用模型并解析为结构化结果。

        返回 ``(归一化结果, warnings, meta)``。模型返回非法 JSON 时抛出
        :class:`~stock_analysis.ai_schema.AIResultError`（异常上带有原始返回文本）。
        """
        text, meta = self.complete(system_prompt, user_content)
        meta["raw_response"] = text
        try:
            result, warnings = validate_and_normalize(text, strict=strict)
        except AIResultError as exc:
            message = str(exc)
            if meta.get("finish_reason") == "length":
                message += (
                    "（模型输出因达到 max_tokens={} 上限被截断，finish_reason=length，"
                    "建议调大 AI_MAX_TOKENS 后重试）".format(self.settings.max_tokens)
                )
            logger.error("模型返回内容无法解析为约定的 JSON：%s", message)
            logger.error("模型原始返回（截断显示）：%s", preview_text(text, 500))
            raise AIResultError(
                message,
                raw_response=exc.raw_response or text,
                meta=meta,
            ) from exc
        return result, warnings, meta

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 SDK 响应或 mock 返回值中抽取文本。"""
        if isinstance(response, str):
            return response

        choices = getattr(response, "choices", None)
        if not choices:
            if isinstance(response, dict):
                choices = response.get("choices")
            if not choices:
                raise AIError("AI 返回结果中没有 choices 字段。")
        choice = choices[0]

        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if content is None:
            raise AIError("AI 返回内容为空（可能被安全策略拦截或达到 token 上限）。")
        if isinstance(content, list):
            # 部分兼容实现返回分段内容
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(getattr(part, "text", part)))
            content = "".join(parts)
        if not str(content).strip():
            raise AIError("AI 返回内容为空字符串。")
        return str(content)

    @staticmethod
    def _build_meta(response: Any, attempt: int, elapsed: float) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "attempts": attempt,
            "elapsed_seconds": round(elapsed, 2),
        }
        finish_reason = None
        choices = getattr(response, "choices", None)
        if not choices and isinstance(response, dict):
            choices = response.get("choices")
        if choices:
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is None and isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")
        meta["finish_reason"] = finish_reason
        if finish_reason == "length":
            logger.warning("模型输出因达到 max_tokens 被截断，结果可能不完整")

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is not None:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, field, None)
                if value is None and isinstance(usage, dict):
                    value = usage.get(field)
                if value is not None:
                    meta[field] = value
        return meta

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "LLMClient(model={!r}, base_url={!r}, api_key={})".format(
            self.settings.model, self.settings.base_url, mask_secret(self.settings.api_key)
        )


def dumps_payload(payload: Any) -> str:
    """把上下文序列化为紧凑 JSON（便于控制 token 数量）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
