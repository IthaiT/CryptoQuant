"""
Universal Raw Data Downloader for Binance
Downloads public aggTrades and Klines data from Binance official data sources.
Supports streaming Zstandard compression for memory efficiency.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging
from datetime import datetime, timedelta
import zstandard as zstd
import requests
from io import BytesIO
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize downloader.
        
        Args:
            base_dir: Base directory for storing data (default: project_root/data/raw_data)
        """
        self.base_dir = base_dir or self._get_default_base_dir()
        self.logger = self._setup_logger()
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory: project_root/data/raw_data"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data" / "raw_data"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging with clean format"""
        # Remove existing handlers
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        
        # Create new handler with clean format
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return logger
  
    def _download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str,  # 'aggTrades' or 'klines'
        interval: Optional[str] = None,  # Required for klines
        custom_header: Optional[List[str]] = None,  # For klines table header
        max_workers: int = 5,  # Number of parallel downloads
    ) -> None:
        """
        Generic download routine for all data types with multi-threading support.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            data_type: Type of data ('aggTrades' or 'klines')
            interval: Kline interval (e.g., '1m') if data_type=='klines'
            custom_header: Custom column names if provided
            max_workers: Number of parallel downloads (default: 5)
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Build description
        data_desc = f"{symbol} {data_type}"
        if interval:
            data_desc += f" ({interval})"
        
        # Print header
        date_range = f"{start_date} 至 {end_date}"
        print(f"\n📥 下载数据: {data_desc} | {date_range}")
        print("-" * 70)
        
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
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_date = {
                    executor.submit(
                        self._process_daily_file,
                        symbol=symbol,
                        date_str=date_str,
                        data_type=data_type,
                        interval=interval,
                        custom_header=custom_header,
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
                            self.logger.error(f"✗ {date_str}: {type(e).__name__}")
                        finally:
                            pbar.update(1)
                except KeyboardInterrupt:
                    print("\n\n⚠️  检测到 Ctrl+C，正在停止下载...")
                    # Cancel all pending futures
                    for future in future_to_date.keys():
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("✅ 已停止下载")
                    return
        
        # Print summary
        print("-" * 70)
        print(f"✅ 完成: 成功下载 {success_count}/{total_days} 天的数据\n")
    
    def _process_daily_file(
        self,
        symbol: str,
        date_str: str,
        data_type: str,
        interval: Optional[str] = None,
        custom_header: Optional[List[str]] = None,
    ) -> bool:
        """
        Core routine to download, extract, process, and save one day of data.
        
        Args:
            symbol: Trading symbol
            date_str: Date string in 'YYYY-MM-DD' format
            data_type: 'aggTrades' or 'klines'
            interval: Kline interval if data_type=='klines'
            custom_header: Custom column names for CSV
        
        Returns:
            True if successful, False otherwise
        """
        # 1. Build output directory and target path
        year = date_str[:4]  # YYYY
        month = date_str[5:7]  # MM
        
        if data_type == 'aggTrades':
            subdir = Path(symbol) / "aggTrades" / year / month
            csv_filename = f"{symbol}-aggTrades-{date_str}.csv"
            output_file = self.base_dir / subdir / f"{symbol}-aggTrades-{date_str}.csv.zst"
        elif data_type == 'klines':
            subdir = Path(symbol) / "klines" / interval / year / month
            csv_filename = f"{symbol}-{interval}-{date_str}.csv"
            output_file = self.base_dir / subdir / f"{symbol}-{interval}-{date_str}.csv.zst"
        else:
            self.logger.error(f"Unknown data_type: {data_type}")
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
        if data_type == 'aggTrades':
            url = f"{self.AGGTRADES_URL}/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        elif data_type == 'klines':
            url = f"{self.KLINES_URL}/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        else:
            return False
        
        try:
            # 3. Download ZIP
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                # 数据未发布，静默跳过
                return False
            
            if response.status_code != 200:
                self.logger.warning(f"⚠️  {date_str}: HTTP {response.status_code}")
                return False
            
            # 4. Extract and read CSV from ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                if csv_filename not in zf.namelist():
                    self.logger.warning(f"⚠️  {date_str}: CSV 文件未找到")
                    return False
                
                with zf.open(csv_filename) as f:
                    # 5. Read CSV with custom or no header
                    if custom_header:
                        # For klines: read without header, then assign
                        df = pd.read_csv(f, header=None, names=custom_header)
                    else:
                        # For aggTrades: read without header, then assign standard names
                        df = pd.read_csv(
                            f,
                            header=None,
                            names=['trade_id', 'price', 'amount', 'first_trade_id',
                                   'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']
                        )
            
            if df.empty:
                self.logger.warning(f"⚠️  {date_str}: 空数据")
                return False
            
            # 6. Standardize columns (data type conversions)
            df = self._standardize_columns(df, data_type)
            
            # 7. Compress and save using Zstandard
            self._save_compressed(df, output_file)
            
            # Successfully saved, no need to print each file
            return True
        
        except Exception as e:
            self.logger.error(f"✗ {date_str}: {type(e).__name__}")
            return False
    
    def _standardize_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize and validate DataFrame columns based on data type.
        
        Args:
            df: Input DataFrame
            data_type: 'aggTrades' or 'klines'
        
        Returns:
            Standardized DataFrame
        """
        if data_type == 'aggTrades':
            # Convert aggTrades numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(int)
            df['trade_id'] = pd.to_numeric(df['trade_id'], errors='coerce').astype(int)
            df['first_trade_id'] = pd.to_numeric(df['first_trade_id'], errors='coerce').astype(int)
            df['last_trade_id'] = pd.to_numeric(df['last_trade_id'], errors='coerce').astype(int)
            df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
            
            # Calculate trade count
            df['num_trades'] = (df['last_trade_id'] - df['first_trade_id']).astype(int) + 1
            
            # Keep relevant columns
            return df[['timestamp', 'price', 'amount', 'is_buyer_maker',
                       'trade_id', 'first_trade_id', 'last_trade_id', 'num_trades']]
        
        elif data_type == 'klines':
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

    def download_agg_trades(self, symbol: str, start_date: str, end_date: str, max_workers: int = 5) -> None:
        """
        Download aggregated trades (raw tick data) from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_workers: Number of parallel downloads (default: 5)
        """
        self._download_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type='aggTrades',
            interval=None,
            custom_header=None,
            max_workers=max_workers,
        )
    
    def download_klines(self, symbol: str, start_date: str, end_date: str, interval: str = "1m", max_workers: int = 5) -> None:
        """
        Download standard Klines (OHLCV data) from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            max_workers: Number of parallel downloads (default: 5)
        """
        self._download_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type='klines',
            interval=interval,
            custom_header=self.KLINES_HEADER,
            max_workers=max_workers,
        )
  