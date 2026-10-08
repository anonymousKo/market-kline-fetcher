import os
import re
import yfinance as yf
import pandas as pd

# 配置多个标的（支持期货、美股、ETF等）
TARGET_TICKERS = ["GC=F", "MU"]

def clean_ticker_name(ticker: str) -> str:
    """清洗标的代码，避免作为文件名时出现非法字符（如 = 或 :）"""
    return re.sub(r'[\\/*?:"<>|=]', '_', ticker)

def fetch_single_ticker(ticker: str, period="3mo", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"⚠️ 标的 {ticker} 未获取到有效数据")
        return pd.DataFrame()

    # 兼容处理 yfinance 多级索引
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

if __name__ == "__main__":
    output_dir = "kline_data"
    os.makedirs(output_dir, exist_ok=True)

    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "w", encoding="utf-8") as f:
            f.write("# 📊 每日 K 线数据汇总 (最近 60 日)\n\n")

    for ticker in TARGET_TICKERS:
        print(f"正在抓取 {ticker} ...")
        kline_df = fetch_single_ticker(ticker)
        if kline_df.empty:
            continue

        csv_text = kline_df.to_csv(index=False)
        safe_name = clean_ticker_name(ticker)

        # 保存为独立 CSV 文件
        file_path = os.path.join(output_dir, f"{safe_name}_kline.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        # 写入 GitHub Summary，使用 HTML details 标签实现折叠显示
        if github_summary:
            with open(github_summary, "a", encoding="utf-8") as f:
                f.write(f"<details><summary><b>点击展开查看：{ticker}</b></summary>\n\n")
                f.write(f"```csv\n{csv_text}\n```\n\n</details>\n\n")

    print(f"全部标的抓取完成，数据已存入 {output_dir}/ 目录。")
