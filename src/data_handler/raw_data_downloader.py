"""
Universal Raw Data Downloader for Binance
Downloads public aggTrades and Klines data from Binance official data sources.
Supports streaming Zstandard compression for memory efficiency.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
import zstandard as zstd
import requests
from io import BytesIO
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from loguru import logger


class RawDataDownloader:
    """
    Universal downloader for Binance public data (aggTrades and Klines).
    
    Features:
    - Download aggTrades (raw tick data) and Klines (OHLCV data)
    - Support for multiple intervals (1m, 5m, 15m, 30m, 1h, etc.)
    - Stream-based Zstandard compression for memory efficiency
    - Progress bars for all download operations
    """
    
    # Binance Vision base URLs
    AGGTRADES_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
    KLINES_URL = "https://data.binance.vision/data/spot/daily/klines"
    
    # Klines table header (required since Binance CSV has no header)
    KLINES_HEADER = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    
    AGGTRADES_HEADER = [
        'trade_id', 'price', 'amount', 'first_trade_id',
        'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match'
    ]
    
    def __init__(
        self, 
        data_type: str,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: Optional[str] = "5m",
        max_workers: Optional[int] = 5,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize downloader.
        Args:
            data_type: 'aggTrades' or 'klines'
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            max_workers: Number of parallel downloads (default: 5)
            base_dir: Base directory for storing data (default: project_root/data/raw_data)
        """
        self.data_type = data_type
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.max_workers = max_workers
        self.base_dir = base_dir or self._get_default_base_dir()
        if self.data_type == 'klines':
            self.bar_header = self.KLINES_HEADER
        else:
            self.bar_header = self.AGGTRADES_HEADER
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory: project_root/data/raw_data"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data" / "raw_data"
  
    def _process_daily_file(self, date_str: str,) -> bool:
        """
        Core routine to download, extract, process, and save one day of data.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
        Returns:
            True if successful, False otherwise
        """
        # 1. Build output directory and target path
        year = date_str[:4]  # YYYY
        month = date_str[5:7]  # MM
        
        if self.data_type == 'aggTrades':
            subdir = Path(self.symbol) / "aggTrades" / year / month
            csv_filename = f"{self.symbol}-aggTrades-{date_str}.csv"
            output_file = self.base_dir / subdir / f"{self.symbol}-aggTrades-{date_str}.csv.zst"
        elif self.data_type == 'klines':
            subdir = Path(self.symbol) / "klines" / self.interval / year / month
            csv_filename = f"{self.symbol}-{self.interval}-{date_str}.csv"
            output_file = self.base_dir / subdir / f"{self.symbol}-{self.interval}-{date_str}.csv.zst"
        else:
            logger.error(f"Unknown data_type: {self.data_type}")
            return False
        
        # Create directory structure
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean up any incomplete temp files from previous interrupted downloads
        temp_file = output_file.with_suffix('.csv.zst.tmp')
        if temp_file.exists():
            temp_file.unlink()
        
        # Skip if already exists
        if output_file.exists():
            # Silently skip, don't log each existing file
            return True
        
        # 2. Build download URL
        if self.data_type == 'aggTrades':
            url = f"{self.AGGTRADES_URL}/{self.symbol}/{self.symbol}-aggTrades-{date_str}.zip"
        elif self.data_type == 'klines':
            url = f"{self.KLINES_URL}/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date_str}.zip"
        else:
            return False
        
        try:
            # 3. Download ZIP
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                # 数据未发布，静默跳过
                return False
            
            if response.status_code != 200:
                logger.warning(f"⚠️  {date_str}: HTTP {response.status_code}")
                return False
            
            # 4. Extract and read CSV from ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                if csv_filename not in zf.namelist():
                    logger.warning(f"⚠️  {date_str}: CSV 文件未找到")
                    return False
                
                with zf.open(csv_filename) as f:
                    # 5. Read CSV with header
                    df = pd.read_csv(f, header=None, names=self.bar_header)

            if df.empty:
                logger.warning(f"⚠️  {date_str}: 空数据")
                return False
            
            # 6. Standardize columns (data type conversions)
            df = self._standardize_columns(df)
            
            # 7. Compress and save using Zstandard
            self._save_compressed(df, output_file)
            
            # Successfully saved, no need to print each file
            return True
        
        except Exception as e:
            logger.error(f"✗ {date_str}: {type(e).__name__}")
            return False
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and validate DataFrame columns based on data type.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Standardized DataFrame
        """
        if self.data_type == 'aggTrades':
            # Convert aggTrades numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(int)
            df['trade_id'] = pd.to_numeric(df['trade_id'], errors='coerce').astype(int)
            df['first_trade_id'] = pd.to_numeric(df['first_trade_id'], errors='coerce').astype(int)
            df['last_trade_id'] = pd.to_numeric(df['last_trade_id'], errors='coerce').astype(int)
            df['is_buyer_maker'] = (
                df['is_buyer_maker']
                .astype(str)
                .str.lower()
                .map({'true': True, 'false': False, '1': True, '0': False})
            )
            
            # Calculate trade count
            df['num_trades'] = (df['last_trade_id'] - df['first_trade_id']).astype(int) + 1
            
            # Keep relevant columns
            return df[['timestamp', 'price', 'amount', 'is_buyer_maker',
                       'trade_id', 'first_trade_id', 'last_trade_id', 'num_trades']]
        
        elif self.data_type == 'klines':
            # Convert klines numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                           'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert time columns
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype(int)
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce').astype(int)
            
            return df
        
        return df
    
    def _save_compressed(self, df: pd.DataFrame, output_file: Path) -> None:
        """
        Save DataFrame to Zstandard compressed CSV using streaming.
        Uses atomic write (temp file + rename) to prevent partial files.
        
        Args:
            df: DataFrame to save
            output_file: Target file path (.csv.zst)
        """
        # Write to temporary file first
        temp_file = output_file.with_suffix('.csv.zst.tmp')
        
        try:
            cctx = zstd.ZstdCompressor(level=3)  # Balanced compression (faster)
            
            with open(temp_file, 'wb') as f_out:
                with cctx.stream_writer(f_out, closefd=False) as writer:
                    csv_data = df.to_csv(index=False)
                    writer.write(csv_data.encode('utf-8'))
            
            # Atomic rename: only if compression succeeded
            temp_file.replace(output_file)
        
        except Exception as e:
            # Clean up temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def download(self) -> None:
        """
        Generic download routine for all data types with multi-threading support.
        """
        start_dt = datetime.fromisoformat(self.start_date)
        end_dt = datetime.fromisoformat(self.end_date)
        
        # Build description
        data_desc = f"{self.symbol} {self.data_type}"
        if self.data_type == 'klines':
            data_desc += f" ({self.interval})"
        
        # Print header via logger
        date_range = f"{self.start_date} 至 {self.end_date}"
        logger.info(f"\n📥 下载数据: {data_desc} | {date_range}")
        logger.info("-" * 70)
        
        # Generate all dates to process
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            dates.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        
        total_days = len(dates)
        success_count = 0
        lock = threading.Lock()
        
        # Progress bar with improved formatting
        with tqdm(
            total=total_days,
            desc="下载进度",
            unit="天",
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green"
        ) as pbar:
            # Multi-threaded download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_date = {
                    executor.submit(
                        self._process_daily_file,
                        date_str=date_str,
                    ): date_str
                    for date_str in dates
                }
                
                # Process completed tasks
                try:
                    for future in as_completed(future_to_date):
                        try:
                            if future.result():
                                with lock:
                                    success_count += 1
                        except Exception as e:
                            date_str = future_to_date[future]
                            logger.error(f"✗ {date_str}: {type(e).__name__}")
                        finally:
                            pbar.update(1)
                except KeyboardInterrupt:
                    logger.warning("\n\n⚠️  检测到 Ctrl+C，正在停止下载...")
                    # Cancel all pending futures
                    for future in future_to_date.keys():
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    logger.info("✅ 已停止下载")
                    return
        
        # Print summary via logger
        logger.info("-" * 70)
        logger.info(f"✅ 完成: 成功下载 {success_count}/{total_days} 天的数据\n")
