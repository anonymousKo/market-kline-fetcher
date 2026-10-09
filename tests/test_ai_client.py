"""AI 客户端测试。

全部使用注入的 ``completion_fn``，**不发起任何真实网络请求**，因此不会消耗 API 额度。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import openai
import pytest

from stock_analysis import ai_schema
from stock_analysis.ai_client import (
    AIAuthError,
    AIConfigError,
    AIError,
    AIRateLimitError,
    AITimeoutError,
    LLMClient,
)
from stock_analysis.config import AISettings
from tests.test_ai_schema import valid_payload


# ---------------------------------------------------------------------------
# 假的 SDK 响应对象
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 100
        self.completion_tokens = 200
        self.total_tokens = 300


class FakeMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: Any, finish_reason: str = "stop") -> None:
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content: Any, finish_reason: str = "stop") -> None:
        self.choices = [FakeChoice(content, finish_reason)]
        self.usage = FakeUsage()


def completion_returning(content: Any):
    def _fn(**kwargs):
        return FakeResponse(content)

    return _fn


def _http_error(cls, status: int, message: str):
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


def make_settings(**overrides) -> AISettings:
    defaults: Dict[str, Any] = {
        "api_key": "sk-test-abcdefghijklmnop",
        "model": "test-model",
        "max_retries": 2,
        "retry_backoff": 2.0,
        "json_mode": True,
    }
    defaults.update(overrides)
    return AISettings(**defaults)


# ---------------------------------------------------------------------------
# 配置检查
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_config_error():
    settings = make_settings(api_key=None)
    calls: List[Dict[str, Any]] = []

    def fn(**kwargs):
        calls.append(kwargs)
        return FakeResponse("{}")

    client = LLMClient(settings, completion_fn=fn)
    assert client.settings.configured is False
    with pytest.raises(AIConfigError) as excinfo:
        client.complete("system", "user")
    assert "AI_API_KEY" in str(excinfo.value)
    assert calls == []  # 未配置时不应发起任何调用


def test_repr_masks_api_key():
    settings = make_settings()
    client = LLMClient(settings, completion_fn=lambda **_: FakeResponse("{}"))
    text = repr(client)
    assert "sk-test-abcdefghijklmnop" not in text
    assert "sk-t***mnop" in text


# ---------------------------------------------------------------------------
# 正常调用
# ---------------------------------------------------------------------------


def test_complete_json_success():
    payload = valid_payload()
    client = LLMClient(make_settings(), completion_fn=completion_returning(json.dumps(payload, ensure_ascii=False)))
    result, warnings, meta = client.complete_json("sys", "user")

    assert result["trend"]["aligned"] is True
    assert warnings == []
    assert meta["attempts"] == 1
    assert meta["total_tokens"] == 300
    assert meta["finish_reason"] == "stop"
    assert "raw_response" in meta


def test_complete_json_accepts_plain_string_response():
    payload = json.dumps(valid_payload(), ensure_ascii=False)
    client = LLMClient(make_settings(), completion_fn=lambda **_: payload)
    result, _warnings, _meta = client.complete_json("sys", "user")
    assert result["summary"]["technical_state"] == "多头排列"


def test_request_contains_json_mode_and_timeout():
    captured: Dict[str, Any] = {}

    def fn(**kwargs):
        captured.update(kwargs)
        return FakeResponse(json.dumps(valid_payload(), ensure_ascii=False))

    settings = make_settings(timeout=12.5)
    LLMClient(settings, completion_fn=fn).complete_json("sys", "user")
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 12.5
    assert captured["model"] == "test-model"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


def test_empty_content_raises_ai_error():
    client = LLMClient(make_settings(), completion_fn=completion_returning(None))
    with pytest.raises(AIError):
        client.complete("sys", "user")


def test_length_finish_reason_logged_but_returns():
    payload = json.dumps(valid_payload(), ensure_ascii=False)
    client = LLMClient(make_settings(), completion_fn=completion_returning(payload))
    text, meta = client.complete("sys", "user")
    assert text == payload
    assert meta["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 重试与错误处理
# ---------------------------------------------------------------------------


def test_retryable_timeout_then_success():
    sleeps: List[float] = []
    attempts: List[int] = []

    def fn(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))
        return FakeResponse(json.dumps(valid_payload(), ensure_ascii=False))

    client = LLMClient(make_settings(), completion_fn=fn, sleep_fn=sleeps.append)
    result, _warnings, meta = client.complete_json("sys", "user")

    assert result["trend"]["aligned"] is True
    assert len(attempts) == 3
    assert meta["attempts"] == 3
    assert sleeps == [2.0, 4.0]  # 指数退避：backoff * 2^(attempt-1)


def test_non_retryable_auth_error_not_retried():
    calls: List[int] = []

    def fn(**kwargs):
        calls.append(1)
        raise _http_error(openai.AuthenticationError, 401, "invalid api key")

    client = LLMClient(make_settings(), completion_fn=fn)
    with pytest.raises(AIAuthError) as excinfo:
        client.complete("sys", "user")
    assert "AI_API_KEY" in str(excinfo.value)
    assert len(calls) == 1  # 鉴权失败不重试


def test_rate_limit_exhausts_retries():
    calls: List[int] = []
    sleeps: List[float] = []

    def fn(**kwargs):
        calls.append(1)
        raise _http_error(openai.RateLimitError, 429, "rate limited")

    client = LLMClient(make_settings(max_retries=1), completion_fn=fn, sleep_fn=sleeps.append)
    with pytest.raises(AIRateLimitError) as excinfo:
        client.complete("sys", "user")
    assert "限流" in str(excinfo.value)
    assert len(calls) == 2  # 1 次初始 + 1 次重试，不会无限重试
    assert sleeps == [2.0]


def test_no_retry_when_max_retries_zero():
    calls: List[int] = []

    def fn(**kwargs):
        calls.append(1)
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))

    client = LLMClient(make_settings(max_retries=0), completion_fn=fn)
    with pytest.raises(AITimeoutError):
        client.complete("sys", "user")
    assert len(calls) == 1


def test_json_mode_fallback_on_bad_request():
    calls: List[Dict[str, Any]] = []

    def fn(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _http_error(openai.BadRequestError, 400, "response_format not supported")
        return FakeResponse(json.dumps(valid_payload(), ensure_ascii=False))

    client = LLMClient(make_settings(), completion_fn=fn)
    result, _warnings, _meta = client.complete_json("sys", "user")

    assert result["trend"]["aligned"] is True
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]  # 自动降级为普通文本模式


def test_unexpected_exception_is_wrapped():
    def fn(**kwargs):
        raise ValueError("boom")

    client = LLMClient(make_settings(), completion_fn=fn)
    with pytest.raises(AIError):
        client.complete("sys", "user")


# ---------------------------------------------------------------------------
# 非法 JSON
# ---------------------------------------------------------------------------


def test_invalid_json_from_model_raises_ai_result_error():
    client = LLMClient(make_settings(), completion_fn=completion_returning("抱歉，我无法完成该请求。"))
    with pytest.raises(ai_schema.AIResultError):
        client.complete_json("sys", "user")


def test_json_missing_core_section_raises():
    payload = valid_payload()
    payload.pop("summary")
    client = LLMClient(make_settings(), completion_fn=completion_returning(json.dumps(payload, ensure_ascii=False)))
    with pytest.raises(ai_schema.AIResultError):
        client.complete_json("sys", "user")


def test_fenced_json_is_accepted():
    text = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
    client = LLMClient(make_settings(), completion_fn=completion_returning(text))
    result, _warnings, _meta = client.complete_json("sys", "user")
    assert result["volume"]["summary"] == "量价配合"


def test_truncated_output_error_mentions_max_tokens():
    """输出因达到 token 上限被截断时，错误信息必须直接给出可操作的提示。"""
    truncated = json.dumps(valid_payload(), ensure_ascii=False)[:120]
    settings = make_settings(max_tokens=4000)
    client = LLMClient(
        settings, completion_fn=lambda **_: FakeResponse(truncated, finish_reason="length")
    )
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        client.complete_json("sys", "user")

    message = str(excinfo.value)
    assert "max_tokens" in message
    assert "截断" in message
    assert "AI_MAX_TOKENS" in message
    # 元信息必须随异常带出，便于持久化（finish_reason / token 数）
    assert excinfo.value.meta.get("finish_reason") == "length"
    assert excinfo.value.meta.get("total_tokens") == 300
    assert excinfo.value.raw_response == truncated
