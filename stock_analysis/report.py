"""把结构化分析结果渲染为中文 Markdown 报告。

原则：报告中的**数值**全部来自程序计算（``indicators`` / ``key_levels``），
大模型只提供文字解读；未生成 AI 分析时，报告仍会输出完整的指标部分。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 2, suffix: str = "", na: str = "数据不足") -> str:
    if value is None:
        return na
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return na
    text = f"{number:,.{digits}f}"
    return f"{text}{suffix}" if suffix else text


def _bullets(items: Optional[List[Any]], empty: str = "无", indent: int = 0) -> str:
    """渲染无序列表；``indent`` 用于在父级列表项下缩进嵌套内容。"""
    prefix = "  " * max(0, indent)
    if not items:
        return f"{prefix}- {empty}"
    lines: List[str] = []
    for item in items:
        if isinstance(item, dict):
            level = item.get("level")
            basis = item.get("basis") or ""
            source = item.get("source") or ""
            parts: List[str] = []
            # 非正数（例如模型用 0 占位）不视为有效价位
            try:
                numeric = float(level) if level is not None else None
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and numeric > 0:
                parts.append(f"**{_fmt(numeric, 2)}**")
            elif not basis:
                basis = "未提供具体价位"
            if basis:
                parts.append(str(basis))
            if source:
                parts.append(f"（来源：{source}）")
            lines.append(prefix + "- " + " — ".join(parts) if parts else prefix + "-")
        else:
            lines.append(f"{prefix}- {item}")
    return "\n".join(lines)


def _indicator_table(indicators: Dict[str, Any]) -> str:
    ma = indicators.get("ma", {}) or {}
    vs = indicators.get("price_vs_ma_pct", {}) or {}
    rsi = indicators.get("rsi", {}) or {}
    macd = indicators.get("macd", {}) or {}
    atr = indicators.get("atr", {}) or {}
    volume = indicators.get("volume", {}) or {}

    rows = [
        ("收盘价", _fmt(indicators.get("close"))),
        ("MA20", _fmt(ma.get("ma20"))),
        ("MA50", _fmt(ma.get("ma50"))),
        ("MA200", _fmt(ma.get("ma200"))),
        ("价格 vs MA20", _fmt(vs.get("vs_ma20"), 2, "%")),
        ("价格 vs MA50", _fmt(vs.get("vs_ma50"), 2, "%")),
        ("价格 vs MA200", _fmt(vs.get("vs_ma200"), 2, "%")),
        ("RSI(14)", _fmt(rsi.get("rsi14"))),
        ("RSI 状态", str(rsi.get("state") or "数据不足")),
        ("MACD DIF", _fmt(macd.get("dif"), 4)),
        ("MACD DEA", _fmt(macd.get("dea"), 4)),
        ("MACD 柱 (DIF-DEA)", _fmt(macd.get("hist"), 4)),
        ("MACD 状态", "{} / {}".format(macd.get("bias", "未知"), macd.get("hist_trend", "未知"))),
        ("ATR(14)", _fmt(atr.get("atr14"))),
        ("ATR 占价格比", _fmt(atr.get("atr_pct"), 2, "%")),
        ("成交量", _fmt(volume.get("volume"), 0)),
        ("20 日均量", _fmt(volume.get("volume_avg20"), 0)),
        ("量比（今量/20 日均量）", _fmt(volume.get("volume_ratio20"), 2, "x")),
    ]
    lines = ["| 指标 | 数值 |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)

    returns = indicators.get("returns_pct", {}) or {}
    if returns:
        lines.append("")
        lines.append("| 收益率窗口 | 交易日数 | 收益率 |")
        lines.append("| --- | --- | --- |")
        for name, payload in returns.items():
            payload = payload or {}
            lines.append(
                f"| 近 {name} | {payload.get('trading_days', '-')} | {_fmt(payload.get('value'), 2, '%')} |"
            )
    return "\n".join(lines)


def _weekly_block(weekly: Dict[str, Any]) -> str:
    if not weekly or not weekly.get("available"):
        warnings = (weekly or {}).get("warnings") or ["周线数据不可用"]
        return "\n".join(f"- {w}" for w in warnings)

    macd = weekly.get("macd", {}) or {}
    lines = [
        f"- 周线数据截止：{weekly.get('as_of', '-')}（共 {weekly.get('weeks', '-')} 根已结束交易周）",
        f"- 周线趋势：**{weekly.get('trend', '数据不足')}**",
        f"- 周线收盘：{_fmt(weekly.get('close'))}",
        f"- 周线 MA10：{_fmt((weekly.get('ma_fast') or {}).get('ma10'))}，"
        f"MA30：{_fmt((weekly.get('ma_slow') or {}).get('ma30'))}",
        f"- 周线 RSI(14)：{_fmt(weekly.get('rsi14'))}（{weekly.get('rsi_state', '-')}）",
        f"- 周线 MACD：DIF {_fmt(macd.get('dif'), 4)} / DEA {_fmt(macd.get('dea'), 4)} / "
        f"柱 {_fmt(macd.get('hist'), 4)}（{macd.get('bias', '未知')}）",
    ]
    for item in weekly.get("evidence") or []:
        lines.append(f"- 判定依据：{item}")
    for item in weekly.get("structure") or []:
        lines.append(f"- 结构：{item}")

    returns = weekly.get("returns_pct") or {}
    if returns:
        parts = []
        for name, payload in returns.items():
            payload = payload or {}
            parts.append(f"{name}: {_fmt(payload.get('value'), 2, '%')}")
        lines.append("- 周线收益：" + "，".join(parts))

    warnings = weekly.get("warnings") or []
    for item in warnings:
        lines.append(f"- ⚠️ {item}")
    return "\n".join(lines)


def _relative_strength_block(rs: Dict[str, Any]) -> str:
    if not rs or not rs.get("available"):
        notes = (rs or {}).get("warnings") or ["基准数据不可用，无法判断相对强弱"]
        return "\n".join(f"- {n}" for n in notes)

    lines: List[str] = []
    for bench in rs.get("benchmarks") or []:
        windows = bench.get("windows") or {}
        parts = []
        for name, payload in windows.items():
            payload = payload or {}
            parts.append(f"{name} 超额 {_fmt(payload.get('excess'), 2, '%')}")
        lines.append(f"- 基准 **{bench.get('symbol')}**：{'；'.join(parts) if parts else '数据不足'}")
    if rs.get("window_note"):
        lines.append(f"- 口径：{rs['window_note']}")
    for warning in rs.get("warnings") or []:
        lines.append(f"- ⚠️ {warning}")
    return "\n".join(lines)


def _key_levels_block(levels: Dict[str, Any]) -> str:
    if not levels or not levels.get("available"):
        return "- 行情数据不足，无法识别关键价位"

    lines: List[str] = []
    extremes = levels.get("recent_extremes") or {}
    for name, payload in extremes.items():
        payload = payload or {}
        lines.append(
            "**{} 区间**（{} 根 K 线）：最高 {}（{}），最低 {}（{}）".format(
                name,
                payload.get("bars_used", "-"),
                _fmt(payload.get("high")),
                payload.get("high_date") or "-",
                _fmt(payload.get("low")),
                payload.get("low_date") or "-",
            )
        )

    if levels.get("insufficient_evidence"):
        lines.append("")
        lines.append("> 暂无足够证据识别可靠的支撑/阻力区域，以下仅列出程序识别到的参考区间。")

    lines.append("")
    lines.append("**阻力区域**（程序聚类结果）")
    lines.append(_bullets(levels.get("resistance_zones"), "未识别到有效阻力区域"))
    lines.append("")
    lines.append("**支撑区域**（程序聚类结果）")
    lines.append(_bullets(levels.get("support_zones"), "未识别到有效支撑区域"))

    if levels.get("breakout_notes"):
        lines.append("")
        lines.append("**突破位置**")
        lines.extend(f"- {item}" for item in levels["breakout_notes"])

    distances = levels.get("ma_distances_pct") or {}
    if distances:
        lines.append("")
        lines.append(
            "**与均线距离**：MA20 {}，MA50 {}，MA200 {}".format(
                _fmt(distances.get("ma20"), 2, "%"),
                _fmt(distances.get("ma50"), 2, "%"),
                _fmt(distances.get("ma200"), 2, "%"),
            )
        )
    if levels.get("cluster_rule"):
        lines.append(f"（区域聚类规则：{levels['cluster_rule']}）")
    lines.append(
        "（分类规则：区域中心价高于现价记为阻力、低于现价记为支撑；"
        "「局部高点/低点」只表示枢轴的来源类型，高点被跌破后可转为支撑）"
    )
    return "\n".join(lines)


def _ai_section(ai: Dict[str, Any]) -> str:
    trend = ai.get("trend", {}) or {}
    momentum = ai.get("momentum", {}) or {}
    divergence = momentum.get("divergence", {}) or {}
    volume = ai.get("volume", {}) or {}
    rs = ai.get("relative_strength", {}) or {}
    levels = ai.get("key_levels", {}) or {}
    scenarios = ai.get("scenarios", {}) or {}
    summary = ai.get("summary", {}) or {}

    parts: List[str] = []
    parts.append("## 一、当前趋势（AI 解读）\n")
    parts.append(f"- **周线趋势**：{trend.get('weekly_trend') or '未提供'}")
    parts.append(f"- **日线趋势**：{trend.get('daily_trend') or '未提供'}")
    parts.append(f"- **多周期是否一致**：{'一致' if trend.get('aligned') else '不一致或无法判断'}")
    parts.append(f"- 说明：{trend.get('alignment_comment') or '未提供'}")
    parts.append("- 判断依据：")
    parts.append(_bullets(trend.get("evidence"), indent=1))

    parts.append("\n## 二、动量分析（AI 解读）\n")
    parts.append(f"- **RSI 状态**：{momentum.get('rsi_state') or '未提供'}")
    parts.append(f"- **MACD 状态**：{momentum.get('macd_state') or '未提供'}")
    parts.append(f"- **动量方向**：{momentum.get('momentum_direction') or '未提供'}")
    parts.append(
        "- **背离判断**：{}".format(
            (divergence.get("comment") or "未提供")
            + ("（模型认为有证据支持）" if divergence.get("has_evidence") else "（模型认为暂无足够证据）")
        )
    )

    parts.append("\n## 三、成交量分析（AI 解读）\n")
    parts.append(f"- {volume.get('volume_relation') or '未提供'}")
    parts.append(f"- {volume.get('price_volume_support') or '未提供'}")
    parts.append(f"- 结论：{volume.get('summary') or '未提供'}")

    parts.append("\n## 四、相对强弱（AI 解读）\n")
    parts.append(f"- 基准：{rs.get('benchmark') or '未提供'}")
    parts.append(f"- 比较区间：{rs.get('period') or '未提供'}")
    parts.append(f"- 结论：{rs.get('outperforming') or '未提供'}")
    parts.append(f"- 说明：{rs.get('summary') or '未提供'}")

    parts.append("\n## 五、支撑位与阻力位（AI 解读）\n")
    parts.append("**支撑位**")
    parts.append(_bullets(levels.get("support")))
    parts.append("")
    parts.append("**阻力位**")
    parts.append(_bullets(levels.get("resistance")))
    parts.append("")
    parts.append(f"- 说明：{levels.get('notes') or '未提供'}")

    parts.append("\n## 六、三种市场情景（AI 解读）\n")
    parts.append(_scenario_block("看涨情景", scenarios.get("bullish") or {}))
    parts.append(_scenario_block("看跌情景", scenarios.get("bearish") or {}))
    parts.append(_scenario_block("中性情景", scenarios.get("neutral") or {}))

    parts.append("\n## 七、最终摘要（AI 解读）\n")
    parts.append(f"- **当前技术状态**：{summary.get('technical_state') or '未提供'}")
    parts.append("- **最值得关注的信号**：")
    parts.append(_bullets(summary.get("top_signals"), indent=1))
    parts.append("- **主要风险**：")
    parts.append(_bullets(summary.get("main_risks"), indent=1))
    parts.append("- **后续观察条件**：")
    parts.append(_bullets(summary.get("watch_conditions"), indent=1))
    parts.append(f"\n> ⚠️ {ai.get('disclaimer') or '技术分析具有局限性，本文不构成任何投资建议。'}")
    return "\n".join(parts)


def _scenario_block(title: str, scenario: Dict[str, Any]) -> str:
    lines = [f"### {title}", "**支持条件**", _bullets(scenario.get("conditions")), "", "**值得观察**", _bullets(scenario.get("watch")), "", "**失效条件**", _bullets(scenario.get("invalidation")), ""]
    return "\n".join(lines)


def render_markdown(record: Dict[str, Any]) -> str:
    """把一条分析记录渲染成 Markdown 报告。"""
    meta = record.get("meta", {}) or {}
    ai = record.get("ai_result")
    quality = record.get("quality", {}) or {}
    indicators = record.get("indicators", {}) or {}
    weekly = record.get("weekly_summary", {}) or {}
    levels = record.get("key_levels", {}) or {}
    rs = record.get("relative_strength", {}) or {}

    symbol = record.get("symbol", "-")
    lines: List[str] = []
    lines.append(f"# {symbol} 技术分析报告")
    lines.append("")
    lines.append(f"- **分析日期**：{record.get('analysis_date', '-')}")
    lines.append(f"- **数据截止**：{quality.get('last_bar_date') or '-'}（时区：{meta.get('timezone') or '-'}）")
    lines.append(f"- **市场 / 币种**：{record.get('market', '-')} / {record.get('currency', '-')}")
    lines.append(f"- **复权口径**：{quality.get('adjustment_note') or '-'}")
    lines.append(f"- **分析模型**：{record.get('model') or '未调用（未配置 API 密钥）'}")
    lines.append(f"- **Prompt 版本**：{record.get('prompt_version', '-')}")
    lines.append(f"- **生成时间（UTC）**：{meta.get('generated_at_utc', '-')}")
    lines.append("")

    lines.append("## 数据质量\n")
    issues = quality.get("issues") or []
    warnings = quality.get("warnings") or []
    if issues:
        lines.append("**严重问题**")
        lines.extend(f"- ❌ {item}" for item in issues)
        lines.append("")
    if warnings:
        lines.append("**提示**")
        lines.extend(f"- ⚠️ {item}" for item in warnings)
        lines.append("")
    if not issues and not warnings:
        lines.append("- 数据质量检查未发现异常。")
    lines.append(f"- 日线 K 线数量：{quality.get('bars', '-')}，区间：{quality.get('start_date', '-')} ~ {quality.get('end_date', '-')}")
    indicator_warnings = ((record.get("context") or {}).get("data_quality") or {}).get("indicator_warnings") or []
    if indicator_warnings:
        lines.append("")
        lines.append("**指标可用性提示**")
        lines.extend(f"- ⚠️ {item}" for item in indicator_warnings)

    lines.append("\n## 指标摘要（程序计算）\n")
    lines.append(_indicator_table(indicators))

    lines.append("\n## 周线趋势摘要（程序计算）\n")
    lines.append(_weekly_block(weekly))

    lines.append("\n## 关键价位（程序计算）\n")
    lines.append(_key_levels_block(levels))

    lines.append("\n## 相对强弱（程序计算）\n")
    lines.append(_relative_strength_block(rs))

    lines.append("\n---\n")
    if ai:
        lines.append(_ai_section(ai))
        ai_warnings = record.get("ai_warnings") or []
        if ai_warnings:
            lines.append("\n### 结果校验提示\n")
            lines.extend(f"- ⚠️ {item}" for item in ai_warnings)
    else:
        lines.append("## AI 分析未生成\n")
        reason = record.get("ai_error") or "未配置 API 密钥，已跳过 AI 分析。"
        lines.append(f"- 原因：{reason}")
        lines.append("- 提示：配置 `.env` 中的 `AI_API_KEY`（可参考 `.env.example`）后重新运行即可生成 AI 分析。")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "> 本报告由程序自动生成：数值均为程序计算结果，AI 仅负责文字解读。"
        "技术分析存在固有局限性，本报告不构成任何投资建议。"
    )
    return "\n".join(lines)
