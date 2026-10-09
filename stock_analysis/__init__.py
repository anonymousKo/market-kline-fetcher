"""AI 股票技术分析工具包。

该包在原有 ``fetch_data.py``（基于 yfinance 的 K 线抓取）之上，新增：

* ``kline``      行情获取、清洗、数据质量检查与日线聚合周线
* ``indicators`` 纯 pandas 实现的技术指标（MA / RSI / MACD / ATR / 量能 / 收益率）
* ``key_levels`` 基于规则的近期价格结构（支撑 / 阻力区域）
* ``context``    构建发送给大模型的结构化 JSON 上下文
* ``prompts``    System Prompt 与 Prompt 版本（与动态数据分离）
* ``ai_client``  与大模型 API 的交互（OpenAI 兼容协议、超时与限流处理）
* ``ai_schema``  大模型返回结果的 JSON Schema 校验与归一化
* ``report``     结构化结果 -> 中文 Markdown 报告
* ``storage``    分析记录持久化（JSON）与后续复核
* ``pipeline``   端到端编排
* ``cli``        命令行入口

该包不会修改或破坏原有 ``fetch_data.py`` 的命令行接口与 CSV 输出格式。
"""

__version__ = "1.0.0"
