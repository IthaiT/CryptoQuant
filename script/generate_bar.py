"""
Generate Dollar Bars from Raw Trade Data
Converts raw aggTrades data for 2025 into Dollar Bars.
Saves processed bars to data/preprocess_data directory.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.bar_generator import DollarBar


def main():
    """Generate Dollar Bars for 2025 data."""
    
    # Configuration
    symbol = "BTCUSDT"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    dollar_threshold = 10_000_000.0  # $10M per bar
    
    # Output directory
    output_dir = project_root / "data" / "preprocess_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Dollar Bar Generation - 2023 Data")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Threshold: ${dollar_threshold:,.0f}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    try:
        # Create Dollar Bar generator
        bar_gen = DollarBar(threshold=dollar_threshold)
        
        # Generate bars
        print(f"\n🔄 Generating Dollar Bars...\n")
        df_bars = bar_gen.generate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            show_progress=True
        )
        
        if df_bars.empty:
            print("⚠️  No bars generated!")
            return
        
        # Save to CSV
        output_file = output_dir / f"{symbol}_{(int)(dollar_threshold/1_000_000)}m_dollar_bars_2023.csv"
        df_bars.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully saved {len(df_bars)} bars to:")
        print(f"   {output_file}")
        print(f"\n📊 Bar Statistics:")
        print(f"   Total bars: {len(df_bars)}")
        print(f"   Time range: {df_bars['timestamp'].min()} → {df_bars['timestamp'].max()}")
        print(f"   Avg price: ${df_bars['close'].mean():.2f}")
        print(f"   Avg volume: {df_bars['volume'].mean():.4f} {symbol.replace('USDT', '')}")
        print(f"   Avg trades per bar: {df_bars['num_trades'].mean():.0f}")
        print(f"   Avg dollar volume per bar: ${df_bars['dollar_volume'].mean():,.0f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
