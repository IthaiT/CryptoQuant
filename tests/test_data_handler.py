"""
Test Pipeline - New Data Pipeline
Tests RawDataDownloader and BarGenerator with streaming decompression and cross-day support.
"""
import sys
from pathlib import Path

from src.data_handler.raw_data_downloader import RawDataDownloader
from src.data_handler.bar_generator import DollarBar, VolumeBar, TickBar
from src.utils.logger import logger

project_root = Path(__file__).parent.parent
# ============================================================
# Configuration
# ============================================================
# Skip download if data already exists locally
SKIP_DOWNLOAD = False  # Set to True to skip download, test bar generation only

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_START_DATE = "2026-01-20"
DEFAULT_END_DATE = "2026-01-26"
BAR_OUTPUT_DIR = project_root / "data" / "tmp_data"


def save_bars(df, filename: str):
    """Save generated bars to data/tmp_data for later analysis."""
    if df is None or df.empty:
        logger.warning("No bars to save for {}", filename)
        return
    BAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BAR_OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info("Saved bars to: {}", output_path)


def test_raw_download():
    """Test downloading raw data from Binance Public Data repository."""
    if SKIP_DOWNLOAD:
        logger.info("=" * 70)
        logger.info("TEST 1: Download Skipped (SKIP_DOWNLOAD=True)")
        logger.info("=" * 70)
        logger.info("To test download, set SKIP_DOWNLOAD=False")
        return
    
    logger.info("=" * 70)
    logger.info("TEST 1: Download Raw Data (AggTrades) from Binance Public Data")
    logger.info("=" * 70)
    logger.info("Downloading aggTrades (Zstandard compressed)...")
    
    downloader = RawDataDownloader()
    
    # Download recent data (use dates when Binance has data)
    try:
        downloader.download_agg_trades(
            symbol=DEFAULT_SYMBOL,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE
        )
        logger.info("Download completed!")
        logger.info("Data saved to: data/raw_data/{}/aggTrades/2026-01/*.csv.zst", DEFAULT_SYMBOL)
    except Exception as e:
        logger.error("Download failed: {}", e)
        logger.info("Tip: Check if dates are available in Binance public data repository")
        raise


def test_generate_dollar_bars():
    """Test Dollar Bar generation (cross-day support)."""
    logger.info("=" * 70)
    logger.info("TEST 2: Generate Dollar Bars (Cross-Day Support)")
    logger.info("=" * 70)
    
    symbol = DEFAULT_SYMBOL
    start_date = DEFAULT_START_DATE
    end_date = DEFAULT_END_DATE
    
    try:
        # Generate Dollar Bars - bars will span across date boundaries
        bar_gen = DollarBar(threshold=500_000.0)  # $500K per bar
        df_bars = bar_gen.generate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Generated {} Dollar Bars", len(df_bars))
        logger.info("Bar Structure:")
        logger.info("  Total bars: {}", len(df_bars))
        if len(df_bars) > 0:
            logger.info("  Time range: {} → {}", df_bars['timestamp'].iloc[0], df_bars['timestamp'].iloc[-1])
            logger.info("  Avg price: ${:.2f}", df_bars['close'].mean())
            logger.info("  Avg volume: {:.4f} BTC", df_bars['volume'].mean())
            logger.info("  Avg trades per bar: {:.0f}", df_bars['num_trades'].mean())
        
        logger.info(
            "First 5 bars:\n{}",
            df_bars[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']].head().to_string(),
        )
        save_bars(df_bars, f"{symbol}_dollar_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        logger.error("Dollar Bar generation failed: {}", e)
        raise


def test_generate_volume_bars():
    """Test Volume Bar generation (cross-day support)."""
    logger.info("=" * 70)
    logger.info("TEST 3: Generate Volume Bars (Cross-Day Support)")
    logger.info("=" * 70)
    
    symbol = DEFAULT_SYMBOL
    start_date = DEFAULT_START_DATE
    end_date = DEFAULT_END_DATE
    
    try:
        # Generate Volume Bars - bars will span across date boundaries
        bar_gen = VolumeBar(threshold=50.0)  # 50 BTC per bar
        df_bars = bar_gen.generate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Generated {} Volume Bars", len(df_bars))
        logger.info("Bar Statistics:")
        logger.info("  Total bars: {}", len(df_bars))
        if len(df_bars) > 0:
            logger.info("  Time range: {} → {}", df_bars['timestamp'].iloc[0], df_bars['timestamp'].iloc[-1])
            logger.info("  Avg volume per bar: {:.4f} BTC", df_bars['volume'].mean())
            logger.info("  Avg price: ${:.2f}", df_bars['close'].mean())
            logger.info("  Avg trades per bar: {:.0f}", df_bars['num_trades'].mean())
        
        logger.info(
            "First 5 bars:\n{}",
            df_bars[['timestamp', 'close', 'volume', 'dollar_volume', 'num_trades']].head().to_string(),
        )
        save_bars(df_bars, f"{symbol}_volume_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        logger.error("Volume Bar generation failed: {}", e)
        raise


def test_generate_tick_bars():
    """Test Tick Bar generation (cross-day support)."""
    logger.info("=" * 70)
    logger.info("TEST 4: Generate Tick Bars (Cross-Day Support)")
    logger.info("=" * 70)
    
    symbol = DEFAULT_SYMBOL
    start_date = DEFAULT_START_DATE
    end_date = DEFAULT_END_DATE
    
    try:
        # Generate Tick Bars - one bar per N trades
        bar_gen = TickBar(threshold=1000)  # 1000 trades per bar
        df_bars = bar_gen.generate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Generated {} Tick Bars (1000 trades/bar)", len(df_bars))
        logger.info("Bar Statistics:")
        logger.info("  Total bars: {}", len(df_bars))
        if len(df_bars) > 0:
            logger.info("  Time range: {} → {}", df_bars['timestamp'].iloc[0], df_bars['timestamp'].iloc[-1])
            logger.info("  Avg trades per bar: {:.0f}", df_bars['num_trades'].mean())
            logger.info("  Avg volume: {:.4f} BTC", df_bars['volume'].mean())
            logger.info("  Avg price: ${:.2f}", df_bars['close'].mean())
        
        logger.info(
            "First 5 bars:\n{}",
            df_bars[['timestamp', 'open', 'close', 'volume', 'num_trades']].head().to_string(),
        )
        save_bars(df_bars, f"{symbol}_tick_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        logger.error("Tick Bar generation failed: {}", e)
        raise


def test_custom_bar_rules():
    """Test custom bar generation by extending BarGenerator."""
    logger.info("=" * 70)
    logger.info("TEST 5: Custom Bar Rules (User-Defined Logic)")
    logger.info("=" * 70)
    
    # Define custom bar class: close bar if price change > 0.5%
    class PriceChangeBar(DollarBar):
        """Custom bar that closes when price change exceeds threshold."""
        
        def __init__(self, price_change_threshold: float = 0.5, base_dir=None):
            """
            Args:
                price_change_threshold: Price change percentage threshold (default 0.5%)
            """
            # Set a large dollar threshold to avoid closing on dollar alone
            super().__init__(threshold=float('inf'), base_dir=base_dir)
            self.price_change_threshold = price_change_threshold
        
        def should_close(self, bar):
            """Close bar if price changed more than threshold."""
            if bar['open'] == 0:
                return False
            pct_change = abs((bar['close'] - bar['open']) / bar['open']) * 100
            return pct_change > self.price_change_threshold
    
    symbol = DEFAULT_SYMBOL
    start_date = DEFAULT_START_DATE
    end_date = DEFAULT_END_DATE
    
    try:
        # Create custom bar generator
        bar_gen = PriceChangeBar(price_change_threshold=0.5)
        df_bars = bar_gen.generate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Generated {} Custom Bars (price change > 0.5%)", len(df_bars))
        logger.info("Bar Statistics:")
        logger.info("  Total bars: {}", len(df_bars))
        if len(df_bars) > 0:
            logger.info("  Time range: {} → {}", df_bars['timestamp'].iloc[0], df_bars['timestamp'].iloc[-1])
            
            # Calculate price change for each bar
            price_changes = ((df_bars['close'] - df_bars['open']).abs() / df_bars['open'] * 100)
            logger.info("  Avg price change per bar: {:.3f}%", price_changes.mean())
            logger.info("  Avg trades per bar: {:.0f}", df_bars['num_trades'].mean())
        
        cols = ['timestamp', 'open', 'close', 'volume', 'num_trades']
        logger.info("First 5 bars:\n{}", df_bars[cols].head().to_string())
        save_bars(df_bars, f"{symbol}_custom_price_change_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        logger.error("Custom Bar generation failed: {}", e)
        raise


def test_klines_download():
    """Test downloading Klines (OHLCV) data from Binance Public Data repository."""
    if SKIP_DOWNLOAD:
        logger.info("=" * 70)
        logger.info("TEST 1.5: Klines Download Skipped (SKIP_DOWNLOAD=True)")
        logger.info("=" * 70)
        logger.info("To test Klines download, set SKIP_DOWNLOAD=False")
        return
    
    logger.info("=" * 70)
    logger.info("TEST 1.5: Download Klines (OHLCV) Data from Binance Public Data")
    logger.info("=" * 70)
    logger.info("Downloading Klines with 1m interval (Zstandard compressed)...")
    
    downloader = RawDataDownloader()
    
    # Download recent data (use dates when Binance has data)
    try:
        downloader.download_klines(
            symbol=DEFAULT_SYMBOL,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
            interval="1m"
        )
        logger.info("Klines download completed!")
        logger.info("Data saved to: data/raw_data/{}/klines/1m/2026-01/*.csv.zst", DEFAULT_SYMBOL)
    except Exception as e:
        logger.error("Klines download failed: {}", e)
        logger.info("Tip: Check if Klines data is available in Binance public data repository")
        raise


def main():
    """Run all tests sequentially."""
    logger.info("=" * 70)
    logger.info("CryptoQuant Data Pipeline Test Suite")
    logger.info("=" * 70)
    
    try:
        # Test 1: Download aggTrades data
        test_raw_download()
        
        # Test 2: Dollar Bars
        test_generate_dollar_bars()
        
        # Test 3: Volume Bars
        test_generate_volume_bars()
        
        # Test 4: Tick Bars
        test_generate_tick_bars()
        
        # Test 5: Custom Bars
        test_custom_bar_rules()
        
        # Test 6: Download Klines data
        test_klines_download()

        
        logger.info("=" * 70)
        logger.info("All tests completed successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("Test Suite Failed: {}", e)
        logger.error("=" * 70)
        logger.exception("Unhandled exception in data pipeline test suite")
        sys.exit(1)


if __name__ == "__main__":
    main()

