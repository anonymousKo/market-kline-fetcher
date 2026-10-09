"""本地模拟大模型服务（OpenAI 兼容），用于验证 AI 流程且**不消耗任何 API 额度**。

用途
----
在没有真实 API 密钥时，验证 "程序 -> openai SDK -> 模型接口 -> JSON 解析 -> 报告"
整条链路是否打通。

用法::

    # 终端 1：启动模拟服务
    python tools/mock_llm_server.py --port 8765

    # 终端 2：把程序指向模拟服务
    AI_API_KEY=sk-mock AI_BASE_URL=http://127.0.0.1:8765/v1 AI_MODEL=mock-model \\
        python main.py analyze --symbol MSFT

返回内容是固定的示例分析（**不是真实分析**），仅用于联调。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

SAMPLE_ANALYSIS = {
    "trend": {
        "weekly_trend": "周线上升（示例数据，非真实分析）",
        "daily_trend": "日线上升（示例数据，非真实分析）",
        "aligned": True,
        "alignment_comment": "两个周期方向一致，属于顺势结构。",
        "evidence": ["周线收盘价高于 MA30", "日线收盘价高于 MA50"],
    },
    "momentum": {
        "rsi_state": "RSI 处于偏强区间，但尚未进入超买，不能单独作为买卖依据。",
        "macd_state": "DIF 位于 DEA 上方，柱状图为正。",
        "momentum_direction": "动量偏强但柱状图增速放缓。",
        "divergence": {"has_evidence": False, "comment": "当前数据不足以支持可靠的背离判断。"},
    },
    "volume": {
        "volume_relation": "当前成交量低于 20 日均量，量比小于 1。",
        "price_volume_support": "上涨未获得成交量配合，突破的有效性需要进一步确认。",
        "summary": "量价存在一定背离，需要放量确认。",
    },
    "relative_strength": {
        "benchmark": "SPY",
        "period": "近 3 个月（63 个交易日）",
        "outperforming": "跑赢",
        "summary": "在相同窗口内个股收益高于基准。",
    },
    "key_levels": {
        "support": [{"level": 0, "basis": "以程序计算的最近支撑区域为准", "source": "程序计算"}],
        "resistance": [{"level": 0, "basis": "以程序计算的最近阻力区域为准", "source": "程序计算"}],
        "notes": "具体价位请以报告中「关键价位（程序计算）」一节为准。",
    },
    "scenarios": {
        "bullish": {
            "conditions": ["放量站上最近阻力区域", "周线维持 MA30 上方"],
            "watch": ["缩量回踩不破 MA20", "MACD 柱状图重新走强"],
            "invalidation": ["跌破最近支撑区域且成交量放大"],
        },
        "bearish": {
            "conditions": ["跌破最近支撑区域", "MACD 出现死叉"],
            "watch": ["反弹无法收复 MA50", "周线跌破 MA30"],
            "invalidation": ["重新站上 MA20 并放量"],
        },
        "neutral": {
            "conditions": ["成交量持续萎缩", "价格在支撑与阻力之间反复"],
            "watch": ["区间边界被有效突破", "波动率（ATR）显著放大"],
            "invalidation": ["出现明确的趋势性放量突破"],
        },
    },
    "summary": {
        "technical_state": "中长期趋势偏多，短期动能边际减弱，量价配合不佳。",
        "top_signals": ["周线与日线趋势一致向上", "价格显著高于 MA200"],
        "main_risks": ["成交量不足导致突破失败", "RSI 处于偏强区间，回调概率上升"],
        "watch_conditions": ["能否放量突破最近阻力区域", "周线 MA30 是否被跌破"],
    },
    "disclaimer": "本内容为模拟服务返回的示例文本，仅用于程序联调，不构成任何投资建议。",
}


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - 遵循 http.server 命名
        length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(length) if length else b""
        payload = {
            "id": "mock-completion",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(SAMPLE_ANALYSIS, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print("[mock-llm] " + format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容的本地模拟服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockHandler)
    print(f"模拟大模型服务已启动：http://{args.host}:{args.port}/v1（仅用于联调，Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
