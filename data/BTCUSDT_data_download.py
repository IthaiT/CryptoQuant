import ccxt
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
import time
import logging
import os

# =====================
# 日志设置
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================
# 代理设置
# =====================
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# =====================
# 全局参数
# =====================
SYMBOL = "BTC/USDT"
# download informer paper dataset
# START_DATE = "2019-08-21"
# END_DATE = "2024-07-24"

START_DATE = "2024-07-25"
END_DATE = "2025-12-25"

LIMIT = 1000

# =====================
# timeframe → 分钟 映射
# =====================
TIMEFRAME_CONFIG = {
#    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
#    "1h": 60,
#    "1d": 1440,
}

# =====================
# 初始化交易所
# =====================
exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 30000,
    "proxies": PROXIES,
    "options": {
        "defaultType": "spot",
    },
})

# =====================
# 时间处理（UTC）
# =====================
def get_time_range(start_date: str, end_date: str):
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(
        tzinfo=timezone.utc,
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )
    return (
        int(start_dt.timestamp() * 1000),
        int(end_dt.timestamp() * 1000),
    )

# =====================
# 主下载函数
# =====================
def download_ohlcv(timeframe: str):
    assert timeframe in TIMEFRAME_CONFIG, f"不支持的周期: {timeframe}"

    tf_minutes = TIMEFRAME_CONFIG[timeframe]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, f"btc-usdt-{timeframe}.csv")

    start_ts, end_ts = get_time_range(START_DATE, END_DATE)
    since = start_ts
    all_bars = []

    total_minutes = (end_ts - start_ts) / (1000 * 60)
    expected_bars = int(total_minutes / tf_minutes)

    logging.info("=" * 60)
    logging.info(f"开始下载 {SYMBOL} {timeframe}")
    logging.info(f"时间范围: {START_DATE} → {END_DATE}")
    logging.info(f"预计 K 线数量: {expected_bars}")
    logging.info("=" * 60)

    with tqdm(total=expected_bars, desc=f"{timeframe} 下载进度", unit="K线") as pbar:
        while since < end_ts:
            try:
                for retry in range(5):
                    try:
                        bars = exchange.fetch_ohlcv(
                            SYMBOL,
                            timeframe=timeframe,
                            since=since,
                            limit=LIMIT
                        )
                        break
                    except Exception as e:
                        if retry < 4:
                            wait = 2 * (retry + 1)
                            logging.warning(f"请求失败，{wait}s 后重试: {e}")
                            time.sleep(wait)
                        else:
                            raise e
            except Exception as e:
                logging.error(f"下载失败: {e}")
                break

            if not bars or bars[-1][0] <= since:
                break

            for bar in bars:
                if bar[0] >= end_ts:
                    break
                all_bars.append(bar)

            since = bars[-1][0] + 1
            pbar.update(len(bars))

            if all_bars:
                last_time = datetime.fromtimestamp(
                    all_bars[-1][0] / 1000,
                    tz=timezone.utc
                )
                pbar.set_postfix({
                    "最新时间": last_time.strftime("%Y-%m-%d %H:%M"),
                    "已下载": len(all_bars)
                })

            time.sleep(exchange.rateLimit / 1000)

    if not all_bars:
        logging.error("没有下载到任何数据")
        return

    # =====================
    # 构建 DataFrame（Binance 风格）
    # =====================
    df = pd.DataFrame(
        all_bars,
        columns=[
            "Open time",
            "Open price",
            "High price",
            "Low price",
            "Close price",
            "Volume",
        ],
    )

    df["Open time"] = df["Open time"].astype("int64")
    df["Close time"] = df["Open time"] + tf_minutes * 60 * 1000 - 1

    df["Quote asset volume"] = 0.0
    df["Number of trades"] = 0
    df["Taker buy base asset volume"] = 0.0
    df["Taker buy quote asset volume"] = 0.0
    df["Ignore"] = 0.0

    df = df[
        [
            "Open time",
            "Open price",
            "High price",
            "Low price",
            "Close price",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ]
    ]

    df = (
        df.drop_duplicates(subset="Open time")
          .sort_values("Open time")
          .reset_index(drop=True)
    )

    numeric_cols = [
        "Open price", "High price", "Low price", "Close price",
        "Volume", "Quote asset volume",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df.to_csv(output_csv, index=False)

    logging.info(f"✅ {timeframe} 下载完成：{len(df)} 行 → {output_csv}")

# =====================
# 示例：下载多个周期
# =====================
if __name__ == "__main__":
    # for tp in TIMEFRAME_CONFIG:
    #     download_ohlcv(tp)
    download_ohlcv("5m")
