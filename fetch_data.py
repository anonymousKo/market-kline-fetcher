import os
import re
import argparse
import yfinance as yf
import pandas as pd

def clean_ticker_name(ticker: str) -> str:
    """清理标的代码，避免生成文件名时出现非法字符"""
    return re.sub(r'[\\/*?:"<>|=]', '_', ticker)

def fetch_single_ticker(ticker: str, period="3mo", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"⚠️ 标的 {ticker} 未获取到有效数据")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)

    date_col = "Date" if "Date" in df.columns else "Datetime"
    df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)

    return df.tail(60)

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch K-line data")
    parser.add_argument("--tickers", type=str, default="", help="标的代码列表，英文逗号分隔")
    parser.add_argument("--period", type=str, default="", help="时间跨度 (1mo, 3mo, 6mo, 1y)")
    args = parser.parse_args()

    # 处理 GitHub 定时触发或传空值时的兜底默认逻辑
    tickers_str = args.tickers.strip() if args.tickers else "GC=F,MU,NVDA,SOXQ"
    period_str = args.period.strip() if args.period else "3mo"

    # 切割为数组
    target_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    return target_tickers, period_str

if __name__ == "__main__":
    target_tickers, period = parse_args()
    print(f"本次运行抓取标的: {target_tickers} | 周期: {period}")

    output_dir = "kline_data"
    os.makedirs(output_dir, exist_ok=True)

    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "w", encoding="utf-8") as f:
            f.write(f"# 📊 每日 K 线数据汇总 (周期: {period})\n\n")

    for ticker in target_tickers:
        print(f"正在抓取 {ticker} ...")
        kline_df = fetch_single_ticker(ticker, period=period)
        if kline_df.empty:
            continue

        csv_text = kline_df.to_csv(index=False)
        safe_name = clean_ticker_name(ticker)

        # 保存独立 CSV
        file_path = os.path.join(output_dir, f"{safe_name}_kline.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        # 写入折叠展示卡片
        if github_summary:
            with open(github_summary, "a", encoding="utf-8") as f:
                f.write(f"<details><summary><b>点击展开查看：{ticker}</b></summary>\n\n")
                f.write(f"```csv\n{csv_text}\n```\n\n</details>\n\n")

    print(f"全部标的抓取完成，数据已存入 {output_dir}/ 目录。")
