import os
import yfinance as yf
import pandas as pd

def get_kline(ticker="GC=F", period="3mo", interval="1d"):
    # 抓取数据
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # 兼容 yfinance 多级索引并提取核心列
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)
    
    # 格式化日期
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    
    # 格式化浮点数保留 2 位小数
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)
        
    return df.tail(60) # 截取最近 60 个交易日

if __name__ == "__main__":
    target_ticker = "GC=F"  # 黄金期货代码，可换成 MU 等美股代码
    kline_df = get_kline(target_ticker)
    csv_text = kline_df.to_csv(index=False)
    
    # 控制台打印
    print(csv_text)
    
    # 保存本地文件
    with open("latest_kline.csv", "w", encoding="utf-8") as f:
        f.write(csv_text)
        
    # 直接写入 GitHub Job 页面摘要（方便在网页上一键复制）
    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as f:
            f.write(f"### {target_ticker} 最近 60 日 K 线数据\n\n```csv\n{csv_text}\n```\n")
