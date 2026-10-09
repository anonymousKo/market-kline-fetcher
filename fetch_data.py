"""行情抓取脚本（保持原有命令行接口与 CSV 输出格式）。

原实现直接调用 ``yfinance``；现在改为复用 :mod:`stock_analysis.kline`，
使"行情抓取"与"技术分析"使用**完全相同的取数、复权与清洗口径**。

对外行为保持不变：

* 命令行参数 ``--tickers`` / ``--period``
* 输出目录 ``kline_data/<代码>_kline.csv``
* CSV 列 ``Date,Open,High,Low,Close,Volume``，价格保留 2 位小数
* 每个标的保留最后 60 根 K 线
* 支持 ``GITHUB_STEP_SUMMARY`` 输出折叠卡片
"""

import argparse
import os
import re
from typing import List, Optional

import pandas as pd

from stock_analysis import kline

DEFAULT_TICKERS = "GC=F,MU,NVDA,SOXQ"
DEFAULT_PERIOD = "3mo"
DEFAULT_LIMIT = 60


def clean_ticker_name(ticker: str) -> str:
    """清理标的代码，避免生成文件名时出现非法字符"""
    return re.sub(r'[\\/*?:"<>|=]', "_", ticker)


def fetch_single_ticker(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = "1d",
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """获取单个标的的 K 线数据。

    返回列与原有实现完全一致：``Date``（``YYYY-MM-DD`` 字符串）、
    ``Open/High/Low/Close``（2 位小数）、``Volume``，并保留最后 ``limit`` 根 K 线。
    """
    try:
        raw = kline.fetch_ohlcv(ticker, period=period, interval=interval, auto_adjust=True)
    except Exception as exc:  # 网络/数据源异常时不让整批任务崩掉
        print(f"⚠️ 标的 {ticker} 行情获取失败：{type(exc).__name__}: {exc}")
        return pd.DataFrame()

    df = kline.normalize_ohlcv(raw, ticker)
    if df.empty:
        print(f"⚠️ 标的 {ticker} 未获取到有效数据")
        return pd.DataFrame()

    df = kline.to_csv_frame(df)
    return df.tail(limit).reset_index(drop=True)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Fetch K-line data")
    parser.add_argument("--tickers", type=str, default="", help="标的代码列表，英文逗号分隔")
    parser.add_argument("--period", type=str, default="", help="时间跨度 (1mo, 3mo, 6mo, 1y)")
    args = parser.parse_args(argv)

    # 处理 GitHub 定时触发或传空值时的兜底默认逻辑
    tickers_str = args.tickers.strip() if args.tickers else DEFAULT_TICKERS
    period_str = args.period.strip() if args.period else DEFAULT_PERIOD

    # 切割为数组
    target_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    return target_tickers, period_str


def run_fetch(
    target_tickers: List[str],
    period: str = DEFAULT_PERIOD,
    output_dir: str = "kline_data",
    limit: int = DEFAULT_LIMIT,
    summary_path: Optional[str] = None,
) -> List[str]:
    """抓取并保存多个标的的行情 CSV，返回写出的文件路径列表。"""
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    github_summary = summary_path or os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "w", encoding="utf-8") as f:
            f.write(f"# 📊 每日 K 线数据汇总 (周期: {period})\n\n")

    for ticker in target_tickers:
        print(f"正在抓取 {ticker} ...")
        kline_df = fetch_single_ticker(ticker, period=period, limit=limit)
        if kline_df.empty:
            continue

        csv_text = kline_df.to_csv(index=False)
        safe_name = clean_ticker_name(ticker)

        # 保存独立 CSV
        file_path = os.path.join(output_dir, f"{safe_name}_kline.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        written.append(file_path)

        # 写入折叠展示卡片
        if github_summary:
            with open(github_summary, "a", encoding="utf-8") as f:
                f.write(f"<details><summary><b>点击展开查看：{ticker}</b></summary>\n\n")
                f.write(f"```csv\n{csv_text}\n```\n\n</details>\n\n")

    return written


if __name__ == "__main__":
    target_tickers, period = parse_args()
    print(f"本次运行抓取标的: {target_tickers} | 周期: {period}")

    output_dir = "kline_data"
    run_fetch(target_tickers, period, output_dir=output_dir)

    print(f"全部标的抓取完成，数据已存入 {output_dir}/ 目录。")
