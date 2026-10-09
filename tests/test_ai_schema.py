"""模型返回结果的解析与校验测试。"""

from __future__ import annotations

import json

import pytest

from stock_analysis import ai_schema


def valid_payload():
    return {
        "trend": {
            "weekly_trend": "周线上升",
            "daily_trend": "日线上升",
            "aligned": True,
            "alignment_comment": "两周期一致",
            "evidence": ["收盘价高于 MA50"],
        },
        "momentum": {
            "rsi_state": "偏强(60-70)",
            "macd_state": "DIF 在 DEA 上方",
            "momentum_direction": "增强",
            "divergence": {"has_evidence": False, "comment": "暂无足够证据"},
        },
        "volume": {
            "volume_relation": "量比 1.3",
            "price_volume_support": "上涨得到成交量支持",
            "summary": "量价配合",
        },
        "relative_strength": {
            "benchmark": "SPY",
            "period": "近 3 个月（63 个交易日）",
            "outperforming": "跑赢",
            "summary": "超额收益为正",
        },
        "key_levels": {
            "support": [{"level": 100.0, "basis": "前低", "source": "程序计算"}],
            "resistance": [{"level": 120.0, "basis": "前高", "source": "程序计算"}],
            "notes": "价位来自枢轴聚类",
        },
        "scenarios": {
            "bullish": {"conditions": ["站稳 MA50"], "watch": ["放量突破"], "invalidation": ["跌破前低"]},
            "bearish": {"conditions": ["跌破 MA200"], "watch": ["放量下跌"], "invalidation": ["重新站上 MA50"]},
            "neutral": {"conditions": ["量能萎缩"], "watch": ["区间突破"], "invalidation": ["趋势加速"]},
        },
        "summary": {
            "technical_state": "多头排列",
            "top_signals": ["站上 MA200"],
            "main_risks": ["RSI 偏高"],
            "watch_conditions": ["能否放量"],
        },
        "disclaimer": "技术分析存在局限。",
    }


def test_parse_and_normalize_valid_payload():
    result, warnings = ai_schema.validate_and_normalize(valid_payload())
    assert warnings == []
    assert result["trend"]["aligned"] is True
    assert result["scenarios"]["bullish"]["conditions"] == ["站稳 MA50"]
    assert result["key_levels"]["support"][0]["level"] == 100.0


def test_parse_json_in_markdown_fence():
    text = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
    result, _ = ai_schema.validate_and_normalize(text)
    assert result["summary"]["technical_state"] == "多头排列"


def test_parse_json_with_surrounding_prose():
    text = "好的，以下是分析结果：\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n以上。"
    result, _ = ai_schema.validate_and_normalize(text)
    assert isinstance(result, dict)


def test_parse_invalid_json_raises():
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.validate_and_normalize("这不是 JSON")


def test_parse_json_array_raises():
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.parse_json_response("[1, 2, 3]")


def test_parse_empty_raises():
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.parse_json_response("")
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.parse_json_response(None)


def test_missing_core_section_raises():
    payload = valid_payload()
    payload.pop("scenarios")
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.normalize_result(payload)
    assert "scenarios" in str(excinfo.value)


def test_missing_nested_field_defaults_with_warning():
    payload = valid_payload()
    del payload["trend"]["evidence"]
    del payload["momentum"]["divergence"]
    result, warnings = ai_schema.normalize_result(payload)

    assert result["trend"]["evidence"] == []
    assert result["momentum"]["divergence"] == {"has_evidence": False, "comment": ""}
    joined = " ".join(warnings)
    assert "trend.evidence" in joined
    assert "momentum.divergence" in joined


def test_type_coercion_and_warnings():
    payload = valid_payload()
    payload["trend"]["aligned"] = "true"  # 字符串布尔
    payload["volume"]["summary"] = 123  # 数字 -> 文本
    payload["summary"]["top_signals"] = "单一信号"  # 字符串 -> 数组
    payload["momentum"]["divergence"]["has_evidence"] = "未知"

    result, warnings = ai_schema.normalize_result(payload)
    assert result["trend"]["aligned"] is True
    assert result["volume"]["summary"] == "123"
    assert result["summary"]["top_signals"] == ["单一信号"]
    assert result["momentum"]["divergence"]["has_evidence"] is False
    assert any("has_evidence" in w for w in warnings)


def test_normalize_support_resistance_items():
    payload = valid_payload()
    payload["key_levels"]["support"] = [
        {"level": "101.5", "basis": "前低"},
        99.0,
        "无法量化的说明",
        {"basis": "仅有依据"},
    ]
    result, warnings = ai_schema.normalize_result(payload)
    supports = result["key_levels"]["support"]
    assert supports[0]["level"] == 101.5
    assert supports[1]["level"] == 99.0
    assert supports[2]["basis"] == "无法量化的说明"
    assert supports[3]["level"] is None


def test_extra_top_level_keys_are_ignored():
    payload = valid_payload()
    payload["extra_field"] = {"foo": "bar"}
    result, warnings = ai_schema.normalize_result(payload)
    assert "extra_field" not in result
    assert any("未定义的字段" in w for w in warnings)


def test_strict_mode_raises_on_missing_field():
    payload = valid_payload()
    del payload["trend"]["alignment_comment"]
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.normalize_result(payload, strict=True)


def test_normalize_non_dict_raises():
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.normalize_result(["not", "a", "dict"])


def test_level_list_with_none():
    payload = valid_payload()
    payload["key_levels"]["resistance"] = None
    result, _ = ai_schema.normalize_result(payload)
    assert result["key_levels"]["resistance"] == []


# ---------------------------------------------------------------------------
# 鲁棒性：多候选选择、解包、schema 回显、诊断信息
# ---------------------------------------------------------------------------


def test_parse_prefers_object_containing_core_keys():
    """说明文字里先出现一个花括号对象时，不能把"假对象"当成结果。"""
    text = (
        "先给一个示例结构与真实结果的区别：{\"note\": \"示例\"}。\n"
        "真实结果如下：\n" + json.dumps(valid_payload(), ensure_ascii=False)
    )
    result, _warnings = ai_schema.validate_and_normalize(text)
    assert result["summary"]["technical_state"] == "多头排列"


def test_parse_ignores_trailing_prose_after_json():
    text = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```\n以上为分析结果，仅供参考。"
    result, _ = ai_schema.validate_and_normalize(text)
    assert isinstance(result, dict)


def test_unwrap_single_key_envelope():
    wrapped = {"analysis": valid_payload()}
    result, warnings = ai_schema.validate_and_normalize(wrapped)
    assert result["trend"]["aligned"] is True
    assert any("解包" in w for w in warnings)


def test_unwrap_nested_envelope_two_levels():
    wrapped = {"data": {"result": valid_payload()}}
    result, warnings = ai_schema.validate_and_normalize(wrapped)
    assert result["summary"]["technical_state"] == "多头排列"


def test_json_schema_echo_raises_clear_error():
    """模型把 JSON Schema 定义当成结果返回时的专门提示。"""
    schema_echo = {
        "type": "object",
        "properties": {
            "trend": {"type": "object", "properties": {"weekly_trend": {"type": "string"}}},
            "summary": {"type": "object", "properties": {"technical_state": {"type": "string"}}},
        },
        "required": ["trend", "summary"],
    }
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.normalize_result(schema_echo)
    message = str(excinfo.value)
    assert "JSON Schema" in message
    assert "指令遵循" in message


def test_schema_like_sections_are_not_silently_unwrapped():
    """schema 的 properties 里虽然有核心字段名，但不能被当成分析数据解包。"""
    schema_echo = {
        "type": "object",
        "properties": {
            "trend": {"type": "object", "properties": {}},
            "momentum": {"type": "object", "properties": {}},
        },
    }
    with pytest.raises(ai_schema.AIResultError):
        ai_schema.normalize_result(schema_echo)


def test_missing_core_error_lists_actual_keys():
    """错误信息必须说明模型实际返回了哪些字段，便于定位。"""
    payload = {"趋势": "上升", "动量": "增强"}
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.normalize_result(payload)
    message = str(excinfo.value)
    assert "trend" in message          # 缺失的必需字段
    assert "趋势" in message            # 模型实际返回的字段
    assert "原始返回开头" in message


def test_error_carries_raw_response_text():
    text = "模型这次没有输出 JSON，只说了几句话。"
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.validate_and_normalize(text)
    assert excinfo.value.raw_response == text


def test_error_carries_raw_response_for_missing_core():
    text = "{\"结果\": \"字段名完全不对\"}"
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.validate_and_normalize(text)
    assert excinfo.value.raw_response == text


def test_preview_text_is_single_line():
    text = ai_schema.preview_text("第一行\n第二行\r\n第三行", 100)
    assert "\n" not in text and "\r" not in text
    assert text.startswith("第一行 第二行")


# ---------------------------------------------------------------------------
# 回归：输出被截断时不得误选内部子对象
# ---------------------------------------------------------------------------


def test_truncated_json_reports_truncation_not_inner_sub_object():
    """复现真实故障：模型输出被截断，且内部子对象里也有 summary 键。

    旧实现会退而选择 relative_strength 子对象，报出“缺少必需的顶层字段：
    trend, momentum, ...”，完全误导排查方向。
    """
    truncated = (
        '{"trend":{"weekly_trend":"上升"},'
        '"relative_strength":{"benchmark":"SPY","period":"3m",'
        '"outperforming":"跑赢","summary":"超额+4.53"},'
        '"key_levels":{"support":'
    )
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.parse_json_response(truncated)

    message = str(excinfo.value)
    assert "不完整或格式错误" in message
    assert "缺少必需的顶层字段" not in message
    assert "AI_MAX_TOKENS" in message


def test_truncated_markdown_fenced_json_is_reported():
    payload = json.dumps(valid_payload(), ensure_ascii=False)
    text = "```json\n" + payload[:120]  # 截断且代码块未闭合
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.parse_json_response(text)
    assert "不完整或格式错误" in str(excinfo.value)


def test_prose_with_truncated_json_does_not_pick_sub_object():
    """说明文字 + 被截断的 JSON：不能把内部子对象当作结果。"""
    text = (
        "分析结果如下：\n"
        '{"relative_strength":{"benchmark":"SPY","summary":"超额"}, "trend":{"weekly_'
    )
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.parse_json_response(text)
    assert "无法从模型返回内容中解析出符合预期的分析 JSON" in str(excinfo.value)


def test_complete_json_followed_by_prose_is_accepted():
    text = json.dumps(valid_payload(), ensure_ascii=False) + "\n\n以上为分析结果，仅供参考。"
    result, _ = ai_schema.validate_and_normalize(text)
    assert result["trend"]["aligned"] is True


def test_single_key_envelope_with_schema_markers_not_unwrapped():
    """包裹层里放的是 schema 时要报 schema 错误，而不是当成数据。"""
    payload = {
        "type": "object",
        "properties": {"trend": {"type": "object"}, "summary": {"type": "object"}},
    }
    with pytest.raises(ai_schema.AIResultError) as excinfo:
        ai_schema.normalize_result(payload)
    assert "JSON Schema" in str(excinfo.value)
