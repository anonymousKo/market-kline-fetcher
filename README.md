# market-kline-fetcher

基于 [yfinance](https://github.com/ranaroussi/yfinance) 的股票行情抓取工具，并在此基础上提供
**AI 技术分析**能力：程序负责取数与计算技术指标，大模型只负责**解释**计算结果，
最终输出结构化的中文 Markdown 分析报告。

> ⚠️ 本项目输出的一切内容均为程序计算结果与模型文字解读，**不构成任何投资建议**。

---

## 1. 功能概览

| 功能 | 说明 |
| --- | --- |
| 行情抓取 | 复用原有 `fetch_data.py`（yfinance），输出格式与命令行参数保持兼容 |
| 技术指标 | MA20/50/200、RSI(14)、MACD(12,26,9)、ATR(14)、成交量与量比、1/3/6 个月收益率 |
| 周期分析 | 日线 + **由日线聚合得到的周线**（自动剔除尚未结束的交易周） |
| 关键价位 | 分形枢轴 + 聚类得到的支撑/阻力**区域**，以及近期区间高低点、与均线距离 |
| 相对强弱 | 相对 SPY / QQQ（可配置）在相同交易日窗口下的超额收益 |
| AI 分析 | 兼容 OpenAI Chat Completions 协议的任何服务商，返回固定 JSON Schema 的结构化结果 |
| 报告输出 | 中文 Markdown 报告 + 结构化 JSON 记录（便于后续统计与回测） |
| 历史复核 | `review` 命令计算未来 5/10/20 个交易日收益、MFE/MAE（无未来数据泄漏） |

---

## 2. 目录结构

```
market-kline-fetcher/
├── fetch_data.py            # 原有行情抓取脚本（行为保持不变，内部复用 stock_analysis.kline）
├── main.py                  # 新的统一命令行入口
├── requirements.txt         # 运行时依赖
├── requirements-dev.txt     # 开发/测试依赖（含 pytest）
├── pytest.ini               # 测试配置
├── .env.example             # 配置项示例（不含真实密钥）
├── stock_analysis/          # 新增的分析包
│   ├── config.py            # 配置与密钥读取（环境变量 / .env）
│   ├── kline.py             # 行情获取、规范化、清洗、质量检查、日线聚合周线
│   ├── indicators.py        # 技术指标与收益率
│   ├── key_levels.py        # 关键价位与价格结构
│   ├── context.py           # 组装发送给大模型的结构化 JSON
│   ├── prompts.py           # System Prompt 与 Prompt 版本（与动态数据分离）
│   ├── ai_client.py         # 大模型 API 客户端（超时、限流、重试、降级）
│   ├── ai_schema.py         # 模型返回 JSON 的解析、校验与归一化
│   ├── report.py            # 结构化结果 -> Markdown 报告
│   ├── storage.py           # 分析记录持久化与未来收益复核
│   ├── pipeline.py          # 端到端编排
│   └── cli.py               # 命令行解析
├── tools/
│   └── mock_llm_server.py   # 本地模拟大模型服务（联调用，不消耗 API 额度）
├── tests/                   # 单元测试（不联网、不消耗 API 额度）
├── kline_data/              # 行情 CSV 输出（默认，已被 .gitignore 忽略）
└── reports/                 # 分析报告与结构化记录（默认，已被 .gitignore 忽略）
```

**模块职责单一**：取数 / 计算 / 组织数据 / 调用模型 / 输出 / 存储彼此独立，
任一环节失败都有明确提示，且 AI 失败不会影响行情获取与指标计算。

---

## 3. 环境准备

* Python **3.9+**（本地验证于 3.9.6；GitHub Actions 使用 3.11）
* 可访问行情数据源（yfinance）
* 如需 AI 分析：一个兼容 OpenAI 协议的模型 API 密钥

```bash
cd market-kline-fetcher
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 安装依赖

```bash
pip install -r requirements.txt        # 运行时依赖
pip install -r requirements-dev.txt    # 需要跑测试时
```

新增依赖（相对原项目）：

| 依赖 | 用途 |
| --- | --- |
| `openai>=1.30.0` | 大模型 API 客户端（兼容 OpenAI 协议的任意服务商） |
| `python-dotenv>=1.0.0` | 从 `.env` 读取配置 |
| `pytest>=7.0` | 单元测试（仅 `requirements-dev.txt`） |

> 技术指标**没有**引入额外的 TA 库：pandas 已足够实现 MA/RSI/MACD/ATR，
> 自行实现可以明确算法口径，避免不同库之间的定义差异（详见第 6 节）。

---

## 4. API 密钥配置

密钥**只从环境变量或 `.env` 文件读取**，源码中不存在任何硬编码密钥，日志中密钥会被脱敏
（如 `sk-a***wxyz`）。

```bash
cp .env.example .env
# 编辑 .env，至少填写 AI_API_KEY（以及按需修改 AI_BASE_URL / AI_MODEL）
```

`.env` 已被 `.gitignore` 忽略，不会被提交。

**只配置行情功能（不使用 AI）**：完全不需要 `.env`，直接运行 `fetch` 命令即可。

### 主要配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AI_API_KEY` | 空 | 密钥；也支持 `OPENAI_API_KEY` |
| `AI_BASE_URL` | `https://api.openai.com/v1` | 兼容 OpenAI 协议的服务地址 |
| `AI_MODEL` | `gpt-4o-mini` | 模型名称 |
| `AI_TIMEOUT` | `60` | 单次请求超时（秒） |
| `AI_MAX_RETRIES` | `2` | 额外重试次数（仅对超时/限流/网络错误；0 = 不重试） |
| `AI_TEMPERATURE` | `0.2` | 采样温度 |
| `AI_MAX_TOKENS` | `8000` | 最大输出 token。本报告的字段较多，**设得太小会导致 JSON 被截断** |
| `AI_JSON_MODE` | `true` | 是否请求服务端返回严格 JSON（不支持时自动降级） |
| `BENCHMARKS` | `SPY,QQQ` | 相对强弱基准；留空则跳过 |
| `ANALYSIS_PERIOD` | `2y` | 分析用日线抓取范围 |
| `MIN_DAILY_BARS` | `260` | 指标所需最少日线数量（不足会给出提示） |
| `KLINE_DIR` / `REPORTS_DIR` | `kline_data` / `reports` | 输出目录 |

以 DeepSeek 为例：

```bash
AI_API_KEY=sk-xxxx
AI_BASE_URL=https://api.deepseek.com/v1
AI_MODEL=deepseek-chat
```

---

## 5. 使用方法

### 5.1 命令行

```bash
# 仅获取行情（兼容原有行为，输出 kline_data/<代码>_kline.csv）
python main.py fetch --symbol MSFT
python main.py fetch --tickers MU,NVDA --period 1y

# 获取行情 + 技术指标 + AI 分析，生成报告
python main.py analyze --symbol MSFT

# 保存行情 CSV，并生成分析报告
python main.py run --symbol MSFT

# 只计算技术指标，不调用 AI（不需要密钥）
python main.py analyze --symbol MSFT --no-ai

# 按历史日期分析（只使用该日期之前的数据）
python main.py analyze --symbol MSFT --as-of 2025-06-30

# 用本地 CSV 离线分析（不联网取数）
python main.py analyze --symbol MSFT --csv kline_data/MSFT_kline.csv

# 对历史分析记录做未来收益复核
python main.py review --min-age-days 30

# 原有入口仍然可用
python fetch_data.py --tickers GC=F,MU,NVDA,SOXQ --period 3mo
```

`analyze` / `run` 常用参数：

| 参数 | 说明 |
| --- | --- |
| `--period` | 数据跨度（默认取 `ANALYSIS_PERIOD`，2y，保证 MA200 与 6 个月收益率可用） |
| `--as-of YYYY-MM-DD` | 指定分析日期（只使用该日期之前的数据） |
| `--csv PATH` | 使用本地 CSV 行情，不联网 |
| `--no-ai` | 跳过 AI 分析 |
| `--no-benchmark` / `--benchmarks SPY,QQQ` | 相对强弱基准控制 |
| `--reports-dir` | 报告输出目录 |
| `--no-save` / `--print-report` / `--print-json` | 不落盘 / 打印 Markdown / 打印结构化 JSON |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 5.2 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `1` | 参数或用法错误 |
| `2` | 行情获取失败或数据严重不足 |
| `3` | 行情与指标正常，但 AI 分析未生成（缺密钥 / 接口错误 / 返回非法 JSON） |

> 退出码 3 时报告**仍然会生成**（含完整的程序计算部分），只是缺少 AI 解读。

### 5.3 输出位置

```
reports/<代码>/<分析日期>_<时分秒>.md      # 中文 Markdown 报告
reports/<代码>/<分析日期>_<时分秒>.json    # 完整结构化记录（含 AI 结果与预留的 forward_returns）
kline_data/<代码>_kline.csv                # 行情 CSV（fetch / run 命令）
```

结构化记录包含：股票代码、分析日期、分析时收盘价、技术指标摘要、关键价位、数据质量、
模型名称、Prompt 版本、AI 结构化结果，以及预留的 `forward_returns` 字段（便于后续扩展与统计）。

若 AI 分析失败，记录中还会包含 **`ai_raw_response`**（模型的原始返回文本，最多 20000 字符），
用于排查“模型没按要求输出 JSON”这类问题：

```bash
python -c "
import json, glob
p = sorted(glob.glob('reports/MSFT/*.json'))[-1]
d = json.load(open(p))
print(d['ai_error'][:400])
print('--- 模型原始返回 ---')
print((d.get('ai_raw_response') or '<未调用模型>')[:1500])
"
```

---

## 6. 计算口径说明

### 6.1 数据与复权

* 数据来源：yfinance，固定使用 `auto_adjust=True`（后复权调整后的 OHLC，含分红/拆股调整）。
* **不会**把未复权与复权价格混用；周线由同一份日线聚合，口径一致。
* 日期为交易所本地交易日（`YYYY-MM-DD`），报告中记录数据时区与数据截止日期。
* 数据质量检查：日期排序与去重、OHLC 缺失、非正价格、`High < Low`、负/零成交量、
  历史长度是否足够、数据是否过期、当日是否可能尚未收盘。
  **缺失值不做任何填充**，也不会伪造 K 线。

### 6.2 日线聚合周线

| 字段 | 规则 |
| --- | --- |
| Open | 该周**第一根有效**日 K 线的开盘价 |
| High | 该周所有日 K 线最高价的最大值 |
| Low | 该周所有日 K 线最低价的最小值 |
| Close | 该周**最后一根有效**日 K 线的收盘价 |
| Volume | 该周成交量之和 |
| Date | 该周最后一个交易日的日期（同时保留 `WeekStart` / `WeekEnd` / `Bars`） |

**未结束的交易周**判定：该周最后一根日 K 落在周五，或参考日期已进入更晚的 ISO 周
（覆盖"周五为节假日、本周提前收市"的情况）才视为已完成；否则视为进行中，
默认从周线指标计算中剔除，并在报告中明确提示。

### 6.3 技术指标

| 指标 | 口径 |
| --- | --- |
| SMA(n) | `close.rolling(n).mean()` |
| EMA(n) | `close.ewm(span=n, adjust=False).mean()`（与主流看盘软件一致） |
| RSI(14) | Wilder 平滑：种子为前 14 个真实涨跌幅的算术平均，之后 `avg=(avg*13+x)/14`；`RSI=100-100/(1+RS)`。边界：平均跌幅为 0 且平均涨幅 > 0 时 = 100；两者均为 0（价格不变）时 = 50；平均涨幅为 0 时 = 0 |
| MACD(12,26,9) | `DIF = EMA12 - EMA26`；`DEA = EMA(DIF,9)`；**柱状图 = DIF - DEA**（注意部分库使用 `2*(DIF-DEA)`） |
| ATR(14) | `TR = max(H-L, |H-prevC|, |L-prevC|)`（第一根 K 线无前收盘价，取 `H-L`），再对 TR 做 Wilder 平滑 |
| 成交量 | 当日量、20 日均量、量比 = 当日量 / 20 日均量 |
| 收益率 | 以**交易日数量**定义：1 个月 = 21 日，3 个月 = 63 日，6 个月 = 126 日；`close[-1]/close[-1-n]-1` |

所有指标只使用当前及之前的数据（滚动/递推计算），**不使用未来数据**；预热期输出空值并给出提示。

### 6.4 关键价位

* 近期区间高低点：最近 20 / 60 个交易日。
* 局部高低点：分形枢轴（前后各 2 根 K 线）；最近 2 根 K 线内的枢轴尚未确认，**不予使用**。
* 支撑/阻力区域：把价格相近的枢轴聚类为区域，容差 = `max(0.5*ATR14, 1%*收盘价)`；
  区域中心价 = 组内枢轴价格均值，同时给出价格范围、触及次数与最近触及日期。
* 区域分类：区域中心价高于现价记为阻力，低于现价记为支撑（局部高点被跌破后可转为支撑）。
* 突破位置：当前收盘价高于"前 20 日高点（剔除最近 5 日）"时，该位置作为潜在支撑参考。
* 与均线距离：价格相对 MA20 / MA50 / MA200 的百分比距离。
* 证据不足时明确输出"暂无足够证据"，不会给出不可靠的精确价位。

### 6.5 相对强弱

* 基准默认 SPY、QQQ（可配置），与个股按**交易日对齐**后使用完全相同的窗口比较，
  输出各窗口的个股收益、基准收益与超额收益。
* 基准数据不可用时明确输出"无法判断相对强弱"，**不会虚构**结果。

---

## 7. AI 分析

* **职责边界**：模型只解释程序算出的指标，不重新计算、不编造数据。
* **数据组织**：以结构化 JSON 发送（股票信息、价格与收益率、日线指标、周线摘要、
  成交量、相对强弱、关键价位、最近 30 根日 K 与 12 根周 K、数据质量提示）；
  完整历史数据保留在本地。
* **Prompt 管理**：System Prompt 与动态数据分离存放于 `stock_analysis/prompts.py`，
  每次分析记录 `PROMPT_VERSION`、模型名称、分析日期与数据截止时间。
* **输出校验**：要求模型返回固定 JSON Schema（趋势 / 动量 / 成交量 / 相对强弱 /
  关键价位 / 三种情景 / 摘要 / 免责声明）。程序会校验必填字段、纠正类型偏差、
  处理 Markdown 代码块包裹与非法 JSON，并记录校验提示（`ai_warnings`）。
  报告中所有**数值**均来自程序计算，未经验证的模型输出不会被当作可靠数据。
* **异常处理**：区分鉴权失败、限流、超时、网络异常、请求非法、服务端错误，
  给出中文提示；仅对瞬时错误做有限次指数退避重试（默认最多 2 次，可配置为 0）。
* **不绑定单一服务商**：任何兼容 OpenAI Chat Completions 协议的服务皆可，
  通过 `AI_BASE_URL` / `AI_MODEL` 切换。

### 报告结构（Markdown）

1. 股票信息（代码、市场、币种、分析日期、数据截止、时区、复权口径、模型、Prompt 版本）
2. 数据质量（严重问题 / 提示 / 指标可用性）
3. 指标摘要（程序计算，含收益率窗口表）
4. 周线趋势摘要（程序计算）
5. 关键价位（程序计算）
6. 相对强弱（程序计算）
7. AI 解读：当前趋势 / 动量分析 / 成交量分析 / 相对强弱 / 支撑与阻力 / 三种情景 / 最终摘要
8. 风险提示

---

## 8. 测试

```bash
pip install -r requirements-dev.txt
python -m pytest
```

* 测试**全部使用合成数据与注入的假客户端**：不联网、**不消耗任何模型 API 额度**。
* 覆盖范围：指标计算与无未来数据、日线聚合周线的正确性（含节假日周与未结束周）、
  数据缺失与异常处理、模型返回非法 JSON、API 错误与重试行为、
  分析记录存储与未来收益复核，以及**原有行情抓取功能未被破坏**的回归测试。

如需验证 AI 链路是否打通（同样不消耗额度）：

```bash
# 终端 1
python tools/mock_llm_server.py --port 8765
# 终端 2
AI_API_KEY=sk-mock AI_BASE_URL=http://127.0.0.1:8765/v1 AI_MODEL=mock-model \
    python main.py analyze --symbol MSFT
```

---

## 9. 常见问题排查

| 现象 | 原因 | 处理方式 |
| --- | --- | --- |
| `未配置大模型 API 密钥，已跳过 AI 分析` | 未设置 `AI_API_KEY` | 配置 `.env`；或使用 `--no-ai` 只算指标 |
| `AI 鉴权失败`（退出码 3） | 密钥无效 / 无权限 | 检查密钥与服务地址是否匹配 |
| `AI 接口触发限流(429)` | 调用过于频繁 | 稍后重试、降低频率或调大 `AI_RETRY_BACKOFF` |
| `AI 请求超时` | 网络慢 / 模型响应慢 | 调大 `AI_TIMEOUT` |
| `AI 请求被拒绝` | 模型名或参数不受支持 | 核对 `AI_MODEL`；不支持 JSON 模式时会自动降级 |
| `模型返回结果缺少必需的顶层字段：...` | 模型没按要求使用英文字段名（如改用中文键名）、把结果包在外层对象中，或只返回了部分内容 | 错误信息会列出**模型实际返回的字段**与原始返回开头；查看记录里的 `ai_raw_response`。只有“多包一层”会被自动解包 |
| `模型返回的是 JSON Schema 定义而不是分析结果` | 模型回显了 Schema 而不是实例（`json_object` 模式下常见） | 换指令遵循更强的模型，或降低 `AI_TEMPERATURE` 后重试 |
| `无法从模型返回内容中解析出 JSON` | 输出被截断 / 夹了大量说明文字 | 调大 `AI_MAX_TOKENS`；查看 `ai_raw_response` 确认是否被截断（`ai_meta.finish_reason == "length"` 表示达到上限） |
| `模型返回内容不是合法的结构化 JSON` | 上述情况的统称 | 以错误信息中的具体原因为准，并查看 `ai_raw_response` |
| `未获取到有效行情数据`（退出码 2） | 代码不存在 / 已退市 / 网络异常 | 检查代码格式（`MSFT`、`000001.SZ`、`GC=F`）与网络 |
| `历史数据不足`（退出码 2） | 新股或数据跨度太短 | 增大 `--period` |
| 报告中 `MA200 / 6m 收益率 = 数据不足` | 历史长度不足 | 增大 `--period`（默认 2y 通常足够） |
| 报告显示"最近一周尚未结束，已排除" | 当前交易周未结束 | 正常行为，避免把进行中的周当成已完成周线 |
| `urllib3 ... NotOpenSSLWarning` | macOS 系统 Python 使用 LibreSSL | 无害警告，可忽略，或改用 Homebrew 安装的 Python |

---

## 10. 已知限制与后续可扩展项

**已知限制**

* 只使用日线数据，不做实时/盘中行情；未结束的交易周与当日未收盘的数据会被明确标记。
* 数据源为 yfinance（免费接口），可能限流或短暂不可用；失败时会明确报错，
  不会静默产出错误结论。
* yfinance 使用后复权价格，历史价格会随新的分红/拆股而整体调整，因此历史报告的绝对价格
  与后续重新抓取的数据可能略有差异（复核时已按同一批复权数据统一基准价来规避）。
* 相对强弱依赖基准行情，基准不可用时该部分会明确标注为不可用。
* 技术指标未包含 KD、布林带、OBV、资金流等；关键价位仅使用可解释的规则，未使用统计模型。

**后续可扩展（当前版本未实现）**

* 自动判定"看涨/看跌条件是否触发"（`review` 已计算未来收益与 MFE/MAE，但尚未自动匹配情景条件）。
* 批量/定时分析（可在 `.github/workflows/fetch_kline.yml` 中增加分析步骤，
  并把 `AI_API_KEY` 配置为 GitHub Secrets）。
* 多标的横向对比、行业基准选择、回测框架接入。
* 更丰富的输出形式（HTML / 邮件 / 消息推送）。

---

## 11. 免责声明

本项目仅用于技术学习与研究。所有指标、价位与 AI 生成的文字均为程序化分析结果，
不含任何投资建议，也不构成对未来价格的预测。请勿将本工具的输出作为唯一的投资决策依据。
