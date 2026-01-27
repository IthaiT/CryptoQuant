"""
High-Performance Bar Generator with Custom Rules Support
Generates financial bars (Dollar, Volume, Tick, and Custom) from raw trade data.
Supports cross-day bar formation without forcing bar boundaries at day ends.

OOP Architecture:
- BarGenerator: Abstract base class providing data loading and generation pipeline
- DollarBar, VolumeBar, TickBar: Concrete bar generators implementing specific logic
- Each subclass defines how bars are initialized, updated, and closed
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import zstandard as zstd
from tqdm import tqdm
from abc import ABC, abstractmethod


class BarGenerator(ABC):
    """
    Abstract base class for bar generation.
    
    Provides infrastructure for data loading, date range discovery, and core generation pipeline.
    Subclasses implement specific bar types (Dollar, Volume, Tick, etc.) by defining
    how bars are initialized, updated, and closed.
    
    Features:
    - Cross-day bar formation (bars aren't forced to close at day boundaries)
    - Zstandard decompression with streaming support
    - Access to historical bars for dynamic threshold calculation
    - Generic bar generation pipeline
    - Progress tracking with tqdm
    
    Subclasses should implement:
    - init_bar(): Initialize a new bar with the first trade
    - update_bar(): Update bar with incoming trade
    - should_close(): Determine if bar should close
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the bar generator.
        
        Args:
            base_dir: Base directory for raw data (default: project_root/data/raw_data)
        """
        self.base_dir = base_dir or self._get_default_base_dir()
        self.logger = self._setup_logger()
        self.bars: List[Dict[str, Any]] = []  # Historical bars for dynamic threshold calculation
    
    def _get_default_base_dir(self) -> Path:
        """Get default base directory: project_root/data/raw_data"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data" / "raw_data"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )
        return logging.getLogger(__name__)
    
    def _load_raw_data(self, symbol: str, date: str) -> pd.DataFrame:
        """Load raw trade data from Zstandard compressed file (streaming)."""
        year = date[:4]  # YYYY
        month = date[5:7]  # MM
        # Try new directory structure first (aggTrades subdirectory)
        zst_path = self.base_dir / symbol / "aggTrades" / year / month / f"{symbol}-aggTrades-{date}.csv.zst"
        
        if not zst_path.exists():
            raise FileNotFoundError(f"Data file not found: {zst_path}")
        
        # Use streaming decompression to avoid memory explosion
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, 'rb') as f_in:
            # Stream decompression directly to DataFrame (no intermediate full data in memory)
            reader = dctx.stream_reader(f_in)
            df = pd.read_csv(reader)
        
        return df
    
    def _get_date_range(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> List[str]:
        """Get list of available dates in [start_date, end_date]."""
        dates: List[str] = []
        current = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        while True:
            date_str = current.strftime("%Y-%m-%d")
            year = current.strftime("%Y")
            month = current.strftime("%m")
            # Check new directory structure (aggTrades subdirectory)
            zst_path = self.base_dir / symbol / "aggTrades" / year / month / f"{symbol}-aggTrades-{date_str}.csv.zst"
            
            if zst_path.exists():
                dates.append(date_str)
                current += timedelta(days=1)
                if end_dt and current > end_dt:
                    break
            else:
                break
        
        return dates
    
    @abstractmethod
    def init_bar(self, ts: int, price: float, amount: float) -> Dict[str, Any]:
        """
        Initialize a new bar with the first trade.
        
        Args:
            ts: Timestamp of the trade
            price: Trade price
            amount: Trade amount
        
        Returns:
            Dictionary representing the new bar
        """
        pass

    @abstractmethod
    def update_bar(self, bar: Dict[str, Any], ts: int, price: float, amount: float) -> None:
        """
        Update existing bar with a new trade.
        
        Args:
            bar: Current bar being formed
            ts: Timestamp of the trade
            price: Trade price
            amount: Trade amount
        """
        pass

    @abstractmethod
    def should_close(self, bar: Dict[str, Any]) -> bool:
        """
        Determine if the current bar should be closed.
        
        Subclasses can access self.bars to implement dynamic thresholds
        based on historical bars.
        
        Args:
            bar: Current bar being formed
        
        Returns:
            True if bar should close, False otherwise
        """
        pass
    
    def generate(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Generate bars using this generator's rules (cross-day support).
        
        This is the main entry point for bar generation.
        Bars are NOT forced to close at day boundaries.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: Optional end date in 'YYYY-MM-DD' format
            show_progress: Whether to show progress bars
        
        Returns:
            DataFrame with generated bars
        """
        dates = self._get_date_range(symbol, start_date, end_date)
        if not dates:
            self.logger.warning(f"⊘ {symbol} 无可用数据 (从 {start_date} 开始)")
            return pd.DataFrame()
        
        self.bars = []  # Reset bars list
        current_bar = None
        
        # Print header
        date_range = f"{dates[0]} 至 {dates[-1]}" if len(dates) > 1 else dates[0]
        print(f"\n📊 生成 Bars: {symbol} | {date_range}")
        
        date_iter = tqdm(
            dates,
            desc="日期处理进度",
            unit="天",
            disable=not show_progress,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="cyan"
        )
        for date in date_iter:
            try:
                df = self._load_raw_data(symbol, date)
            except FileNotFoundError:
                continue
            
            # Use numpy arrays for speed (vectorized operations)
            timestamps = df['timestamp'].values
            prices = df['price'].astype(float).values
            amounts = df['amount'].astype(float).values
            
            # Process trades without inner progress bar (for performance)
            for ts, price, amount in zip(timestamps, prices, amounts):
                if current_bar is None:
                    current_bar = self.init_bar(ts, price, amount)
                else:
                    self.update_bar(current_bar, ts, price, amount)
                
                # Check if bar should close
                if self.should_close(current_bar):
                    current_bar['close_time'] = ts
                    self.bars.append(current_bar)
                    current_bar = None
        
        result_df = pd.DataFrame(self.bars)
        print(f"✅ 生成完成: {len(result_df)} 个 Bars\n")
        
        return result_df


class DollarBar(BarGenerator):
    """Bar closes when cumulative dollar value reaches threshold."""
    
    def __init__(self, threshold: float, base_dir: Optional[Path] = None) -> None:
        super().__init__(base_dir)
        self.threshold = threshold
        self.cumulative = 0.0

    def init_bar(self, ts: int, price: float, amount: float) -> Dict[str, Any]:
        value = price * amount
        self.cumulative = value
        return {
            'timestamp': ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': amount,
            'dollar_volume': value,
            'num_trades': 1,
        }

    def update_bar(self, bar: Dict[str, Any], ts: int, price: float, amount: float) -> None:
        value = price * amount
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += amount
        bar['dollar_volume'] += value
        bar['num_trades'] += 1
        self.cumulative += value

    def should_close(self, bar: Dict[str, Any]) -> bool:
        # Can access self.bars here for dynamic threshold calculation
        # Example: if len(self.bars) > 30:
        #     self.threshold = calculate_ema_from_bars(self.bars[-30:])
        return self.cumulative >= self.threshold


class VolumeBar(BarGenerator):
    """Bar closes when cumulative volume reaches threshold."""
    
    def __init__(self, threshold: float, base_dir: Optional[Path] = None) -> None:
        super().__init__(base_dir)
        self.threshold = threshold
        self.cumulative = 0.0

    def init_bar(self, ts: int, price: float, amount: float) -> Dict[str, Any]:
        value = price * amount
        self.cumulative = amount
        return {
            'timestamp': ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': amount,
            'dollar_volume': value,
            'num_trades': 1,
        }

    def update_bar(self, bar: Dict[str, Any], ts: int, price: float, amount: float) -> None:
        value = price * amount
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += amount
        bar['dollar_volume'] += value
        bar['num_trades'] += 1
        self.cumulative += amount

    def should_close(self, bar: Dict[str, Any]) -> bool:
        return self.cumulative >= self.threshold


class TickBar(BarGenerator):
    """Bar closes after a fixed number of trades."""
    
    def __init__(self, threshold: int, base_dir: Optional[Path] = None) -> None:
        super().__init__(base_dir)
        self.threshold = threshold
        self.trade_count = 0

    def init_bar(self, ts: int, price: float, amount: float) -> Dict[str, Any]:
        value = price * amount
        self.trade_count = 1
        return {
            'timestamp': ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': amount,
            'dollar_volume': value,
            'num_trades': 1,
        }

    def update_bar(self, bar: Dict[str, Any], ts: int, price: float, amount: float) -> None:
        value = price * amount
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += amount
        bar['dollar_volume'] += value
        bar['num_trades'] += 1
        self.trade_count += 1

    def should_close(self, bar: Dict[str, Any]) -> bool:
        return self.trade_count >= self.threshold
