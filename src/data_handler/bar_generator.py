"""
High-Performance Bar Generator with Custom Rules Support
Generates financial bars (Dollar, Volume, Tick, and Custom) from raw trade data.
Supports cross-day bar formation without forcing bar boundaries at day ends.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import zstandard as zstd
from tqdm import tqdm
from abc import ABC, abstractmethod
from src.data_handler.bar_rule import BarRule


class BarGeneratorUtil:
    """Utility class for data loading and date range discovery."""
    
    @staticmethod
    def get_default_base_dir() -> Path:
        """Get default data directory: project_root/data/raw_data"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data" / "raw_data"
    
    @staticmethod
    def setup_logger() -> logging.Logger:
        """Setup logger."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )
        return logging.getLogger(__name__)
    
    @staticmethod
    def load_raw_data(base_dir: Path, symbol: str, date: str) -> pd.DataFrame:
        """Load raw trade data from Zstandard compressed file with streaming decompression."""
        year = date[:4]
        month = date[5:7]
        zst_path = base_dir / symbol / "aggTrades" / year / month / f"{symbol}-aggTrades-{date}.csv.zst"
        
        if not zst_path.exists():
            raise FileNotFoundError(f"Data file not found: {zst_path}")
        
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, 'rb') as f_in:
            reader = dctx.stream_reader(f_in)
            df = pd.read_csv(reader)
        
        return df
    
    @staticmethod
    def get_date_range(base_dir: Path, symbol: str, start_date: str, end_date: Optional[str] = None) -> List[str]:
        """Get list of available dates in [start_date, end_date]."""
        dates: List[str] = []
        current = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        while True:
            date_str = current.strftime("%Y-%m-%d")
            year = current.strftime("%Y")
            month = current.strftime("%m")
            zst_path = base_dir / symbol / "aggTrades" / year / month / f"{symbol}-aggTrades-{date_str}.csv.zst"
            
            if zst_path.exists():
                dates.append(date_str)
                current += timedelta(days=1)
                if end_dt and current > end_dt:
                    break
            else:
                break
        
        return dates


class BarGenerator:
    """
    Generates various types of bars (Dollar, Volume, Tick) from raw trade data.
    
    Features:
    - Cross-day bar formation (bars aren't forced to close at day boundaries)
    - Streaming Zstandard decompression
    - Strategy pattern: Flexible bar generation logic via BarRule
    - Period-based generation (day/month/year)
    - Memory-efficient: Yields data period-by-period
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize Bar Generator."""
        self.base_dir = base_dir or BarGeneratorUtil.get_default_base_dir()
        self.logger = BarGeneratorUtil.setup_logger()
        self.bars: List[Dict[str, Any]] = []
    
    def generate(
        self,
        bar_rule: BarRule,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Main entry for bar generation. Bars are not forced to close at day boundaries.
        
        Args:
            bar_rule: Bar rule instance
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (optional)
            period: Time period grouping ('day', 'month', 'year', or None)
            show_progress: Whether to show progress bar
        
        Returns:
            DataFrame if period is None, else generator yielding (period_key, DataFrame)
        """
        dates = BarGeneratorUtil.get_date_range(self.base_dir, symbol, start_date, end_date)
        if not dates:
            self.logger.warning(f"⊘ {symbol} no data available from {start_date}")
            if period is None:
                return pd.DataFrame()
            else:
                return
        
        if period is None:
            return self._generate_all(bar_rule, symbol, dates, show_progress)
        
        if period not in ('day', 'month', 'year'):
            raise ValueError(f"Invalid period: {period}. Must be 'day', 'month', 'year', or None")
        
        return self._generate_by_period(bar_rule, symbol, dates, period, show_progress)
    
    def _generate_all(
        self, 
        bar_rule: BarRule,
        symbol: str, 
        dates: List[str], 
        show_progress: bool
    ) -> pd.DataFrame:
        """Generate all bars and return a single DataFrame."""
        self.bars = []
        current_bar = None
        
        date_range = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]
        print(f"\n📊 Generating Bars: {symbol} | {date_range}")
        
        date_iter = tqdm(
            dates,
            desc="Processing dates",
            unit="day",
            disable=not show_progress,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="cyan"
        )
        
        for date in date_iter:
            try:
                df = BarGeneratorUtil.load_raw_data(self.base_dir, symbol, date)
            except FileNotFoundError:
                continue
            
            timestamps = df['timestamp'].values
            prices = df['price'].astype(float).values
            amounts = df['amount'].astype(float).values
            
            for ts, price, amount in zip(timestamps, prices, amounts):
                if current_bar is None:
                    current_bar = bar_rule.init_bar(ts, price, amount)
                else:
                    bar_rule.update_bar(current_bar, ts, price, amount)
                
                if bar_rule.should_close(current_bar):
                    current_bar['close_time'] = ts
                    self.bars.append(current_bar)
                    bar_rule.reset()
                    current_bar = None
        
        result_df = pd.DataFrame(self.bars)
        print(f"✅ Complete: {len(result_df)} bars\n")
        
        return result_df
    
    def _generate_by_period(
        self, 
        bar_rule: BarRule,
        symbol: str, 
        dates: List[str],
        period: str, 
        show_progress: bool
    ):
        """Generate bars grouped by time period. Yields (period_key, DataFrame) for each period."""
        from collections import defaultdict
        
        period_groups = defaultdict(list)
        for date in dates:
            if period == 'day':
                key = date
            elif period == 'month':
                key = date[:7]
            else:  # year
                key = date[:4]
            period_groups[key].append(date)
        
        sorted_periods = sorted(period_groups.keys())
        
        print(f"\n📊 Generating Bars by {period}: {symbol}")
        print(f"   Total periods: {len(sorted_periods)}")
        
        period_iter = tqdm(
            sorted_periods,
            desc=f"Processing {period}s",
            unit=period,
            disable=not show_progress,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="cyan"
        )
        
        for period_key in period_iter:
            self.bars = []
            current_bar = None
            period_dates = period_groups[period_key]
            
            for date in period_dates:
                try:
                    df = BarGeneratorUtil.load_raw_data(self.base_dir, symbol, date)
                except FileNotFoundError:
                    continue
                
                timestamps = df['timestamp'].values
                prices = df['price'].astype(float).values
                amounts = df['amount'].astype(float).values
                
                for ts, price, amount in zip(timestamps, prices, amounts):
                    if current_bar is None:
                        current_bar = bar_rule.init_bar(ts, price, amount)
                    else:
                        bar_rule.update_bar(current_bar, ts, price, amount)
                    
                    if bar_rule.should_close(current_bar):
                        current_bar['close_time'] = ts
                        self.bars.append(current_bar)
                        bar_rule.reset()
                        current_bar = None
            
            if self.bars:
                yield period_key, pd.DataFrame(self.bars)
        
        print(f"✅ Complete\n")


