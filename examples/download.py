"""
Memory-safe Binance Raw Data Downloader
完全流式处理版本（不会 OOM）
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime, timedelta
import zstandard as zstd
import requests
import zipfile
from tqdm import tqdm


class RawDataDownloader:

    AGGTRADES_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
    KLINES_URL = "https://data.binance.vision/data/spot/daily/klines"

    KLINES_HEADER = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]

    AGGTRADES_HEADER = [
        'trade_id', 'price', 'amount', 'first_trade_id',
        'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match'
    ]

    # ⭐ 每块 200k 行，内存极稳
    CHUNK_SIZE = 200_000

    def __init__(
        self,
        data_type: str,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: Optional[str] = "5m",
        base_dir: Optional[Path] = None
    ):
        self.data_type = data_type
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        self.base_dir = base_dir or Path("data/raw_data")
        self.logger = self._setup_logger()

        self.header = (
            self.KLINES_HEADER if data_type == "klines"
            else self.AGGTRADES_HEADER
        )

    # =====================================================
    # 基础设施
    # =====================================================

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    # =====================================================
    # 核心：真正的流式处理（重点）
    # =====================================================

    def _process_daily_file(self, date_str: str) -> bool:

        year = date_str[:4]
        month = date_str[5:7]

        if self.data_type == "aggTrades":
            subdir = Path(self.symbol) / "aggTrades" / year / month
            csv_name = f"{self.symbol}-aggTrades-{date_str}.csv"
            url = f"{self.AGGTRADES_URL}/{self.symbol}/{csv_name.replace('.csv','.zip')}"
        else:
            subdir = Path(self.symbol) / "klines" / self.interval / year / month
            csv_name = f"{self.symbol}-{self.interval}-{date_str}.csv"
            url = f"{self.KLINES_URL}/{self.symbol}/{self.interval}/{csv_name.replace('.csv','.zip')}"

        output_file = self.base_dir / subdir / f"{csv_name}.zst"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            return True

        try:
            # ⭐ streaming 下载（不读入内存）
            with requests.get(url, stream=True, timeout=60) as r:

                if r.status_code != 200:
                    return False

                with zipfile.ZipFile(r.raw) as zf:
                    with zf.open(csv_name) as f_csv:

                        temp_file = output_file.with_suffix(".tmp")

                        cctx = zstd.ZstdCompressor(level=3)

                        with open(temp_file, "wb") as fout:
                            with cctx.stream_writer(fout) as writer:

                                first_chunk = True

                                # ⭐⭐⭐ 核心：chunk 读取 ⭐⭐⭐
                                for chunk in pd.read_csv(
                                        f_csv,
                                        header=None,
                                        names=self.header,
                                        chunksize=self.CHUNK_SIZE
                                ):
                                    chunk = self._standardize(chunk)

                                    # ⭐ 直接写入压缩流（零拷贝）
                                    chunk.to_csv(
                                        writer,
                                        index=False,
                                        header=first_chunk
                                    )
                                    first_chunk = False

                        temp_file.replace(output_file)

            return True

        except Exception:
            return False

    # =====================================================
    # 列标准化（保持轻量）
    # =====================================================

    def _standardize(self, df: pd.DataFrame):

        if self.data_type == "aggTrades":
            df['price'] = pd.to_numeric(df['price'])
            df['amount'] = pd.to_numeric(df['amount'])
            df['timestamp'] = pd.to_numeric(df['timestamp']).astype(int)

        return df

    # =====================================================
    # 下载主循环（单线程 = 稳定王）
    # =====================================================

    def download(self):

        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)

        dates = []
        while start <= end:
            dates.append(start.strftime("%Y-%m-%d"))
            start += timedelta(days=1)

        print(f"\n📥 {self.symbol} {self.data_type} 下载中...\n")

        success = 0

        # ⭐ 单线程稳定模式
        for d in tqdm(dates, desc="下载进度", unit="天"):
            if self._process_daily_file(d):
                success += 1

        print(f"\n✅ 完成 {success}/{len(dates)} 天")

data_type = 'aggTrades' # 'aggTrades' or 'klines'
symbol = "BTCUSDT"
start_date = "2020-01-01"
end_date = "2025-12-31"
# inteval = "5m" # for klines
# base_dir = 'project_root/data/raw_data'

downloader = RawDataDownloader(data_type, symbol, start_date, end_date)
downloader.download()