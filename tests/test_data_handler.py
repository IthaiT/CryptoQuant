"""
Test Pipeline - New Data Pipeline
Tests RawDataDownloader and BarGenerator with streaming decompression and cross-day support.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_handler.raw_data_downloader import RawDataDownloader
from src.data_handler.bar_generator import DollarBar, VolumeBar, TickBar

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
        print(f"⚠️ No bars to save for {filename}")
        return
    BAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BAR_OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"💾 Saved bars to: {output_path}")


def test_raw_download():
    """Test downloading raw data from Binance Public Data repository."""
    if SKIP_DOWNLOAD:
        print("\n" + "=" * 70)
        print("TEST 1: Download Skipped (SKIP_DOWNLOAD=True)")
        print("=" * 70)
        print("💡 To test download, set SKIP_DOWNLOAD=False")
        return
    
    print("\n" + "=" * 70)
    print("TEST 1: Download Raw Data (AggTrades) from Binance Public Data")
    print("=" * 70)
    print("📝 Downloading aggTrades (Zstandard compressed)...\n")
    
    downloader = RawDataDownloader()
    
    # Download recent data (use dates when Binance has data)
    try:
        downloader.download_agg_trades(
            symbol=DEFAULT_SYMBOL,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE
        )
        print("✅ Download completed!")
        print(f"📁 Data saved to: data/raw_data/{DEFAULT_SYMBOL}/aggTrades/2026-01/*.csv.zst")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Tip: Check if dates are available in Binance public data repository")
        raise


def test_generate_dollar_bars():
    """Test Dollar Bar generation (cross-day support)."""
    print("\n" + "=" * 70)
    print("TEST 2: Generate Dollar Bars (Cross-Day Support)")
    print("=" * 70)
    
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
        
        print(f"✅ Generated {len(df_bars)} Dollar Bars\n")
        print("📊 Bar Structure:")
        print(f"  Total bars: {len(df_bars)}")
        if len(df_bars) > 0:
            print(f"  Time range: {df_bars['timestamp'].iloc[0]} → {df_bars['timestamp'].iloc[-1]}")
            print(f"  Avg price: ${df_bars['close'].mean():.2f}")
            print(f"  Avg volume: {df_bars['volume'].mean():.4f} BTC")
            print(f"  Avg trades per bar: {df_bars['num_trades'].mean():.0f}")
        
        print("\n📈 First 5 bars:")
        print(df_bars[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']].head())
        save_bars(df_bars, f"{symbol}_dollar_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        print(f"❌ Dollar Bar generation failed: {e}")
        raise


def test_generate_volume_bars():
    """Test Volume Bar generation (cross-day support)."""
    print("\n" + "=" * 70)
    print("TEST 3: Generate Volume Bars (Cross-Day Support)")
    print("=" * 70)
    
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
        
        print(f"✅ Generated {len(df_bars)} Volume Bars\n")
        print("📊 Bar Statistics:")
        print(f"  Total bars: {len(df_bars)}")
        if len(df_bars) > 0:
            print(f"  Time range: {df_bars['timestamp'].iloc[0]} → {df_bars['timestamp'].iloc[-1]}")
            print(f"  Avg volume per bar: {df_bars['volume'].mean():.4f} BTC")
            print(f"  Avg price: ${df_bars['close'].mean():.2f}")
            print(f"  Avg trades per bar: {df_bars['num_trades'].mean():.0f}")
        
        print("\n📈 First 5 bars:")
        print(df_bars[['timestamp', 'close', 'volume', 'dollar_volume', 'num_trades']].head())
        save_bars(df_bars, f"{symbol}_volume_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        print(f"❌ Volume Bar generation failed: {e}")
        raise


def test_generate_tick_bars():
    """Test Tick Bar generation (cross-day support)."""
    print("\n" + "=" * 70)
    print("TEST 4: Generate Tick Bars (Cross-Day Support)")
    print("=" * 70)
    
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
        
        print(f"✅ Generated {len(df_bars)} Tick Bars (1000 trades/bar)\n")
        print("📊 Bar Statistics:")
        print(f"  Total bars: {len(df_bars)}")
        if len(df_bars) > 0:
            print(f"  Time range: {df_bars['timestamp'].iloc[0]} → {df_bars['timestamp'].iloc[-1]}")
            print(f"  Avg trades per bar: {df_bars['num_trades'].mean():.0f}")
            print(f"  Avg volume: {df_bars['volume'].mean():.4f} BTC")
            print(f"  Avg price: ${df_bars['close'].mean():.2f}")
        
        print("\n📈 First 5 bars:")
        print(df_bars[['timestamp', 'open', 'close', 'volume', 'num_trades']].head())
        save_bars(df_bars, f"{symbol}_tick_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        print(f"❌ Tick Bar generation failed: {e}")
        raise


def test_custom_bar_rules():
    """Test custom bar generation by extending BarGenerator."""
    print("\n" + "=" * 70)
    print("TEST 5: Custom Bar Rules (User-Defined Logic)")
    print("=" * 70)
    
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
        
        print(f"✅ Generated {len(df_bars)} Custom Bars (price change > 0.5%)\n")
        print("📊 Bar Statistics:")
        print(f"  Total bars: {len(df_bars)}")
        if len(df_bars) > 0:
            print(f"  Time range: {df_bars['timestamp'].iloc[0]} → {df_bars['timestamp'].iloc[-1]}")
            
            # Calculate price change for each bar
            price_changes = ((df_bars['close'] - df_bars['open']).abs() / df_bars['open'] * 100)
            print(f"  Avg price change per bar: {price_changes.mean():.3f}%")
            print(f"  Avg trades per bar: {df_bars['num_trades'].mean():.0f}")
        
        print("\n📈 First 5 bars:")
        cols = ['timestamp', 'open', 'close', 'volume', 'num_trades']
        print(df_bars[cols].head())
        save_bars(df_bars, f"{symbol}_custom_price_change_{start_date}_{end_date}.csv")
        
        return df_bars
    except Exception as e:
        print(f"❌ Custom Bar generation failed: {e}")
        raise


def test_klines_download():
    """Test downloading Klines (OHLCV) data from Binance Public Data repository."""
    if SKIP_DOWNLOAD:
        print("\n" + "=" * 70)
        print("TEST 1.5: Klines Download Skipped (SKIP_DOWNLOAD=True)")
        print("=" * 70)
        print("💡 To test Klines download, set SKIP_DOWNLOAD=False")
        return
    
    print("\n" + "=" * 70)
    print("TEST 1.5: Download Klines (OHLCV) Data from Binance Public Data")
    print("=" * 70)
    print("📝 Downloading Klines with 1m interval (Zstandard compressed)...\n")
    
    downloader = RawDataDownloader()
    
    # Download recent data (use dates when Binance has data)
    try:
        downloader.download_klines(
            symbol=DEFAULT_SYMBOL,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
            interval="1m"
        )
        print("✅ Klines download completed!")
        print(f"📁 Data saved to: data/raw_data/{DEFAULT_SYMBOL}/klines/1m/2026-01/*.csv.zst")
    except Exception as e:
        print(f"❌ Klines download failed: {e}")
        print("💡 Tip: Check if Klines data is available in Binance public data repository")
        raise


def main():
    """Run all tests sequentially."""
    print("\n" + "=" * 70)
    print("CryptoQuant Data Pipeline Test Suite")
    print("=" * 70)
    
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

        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Test Suite Failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

