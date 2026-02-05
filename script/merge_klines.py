"""
Merge Klines Data into Single CSV
Combines multiple daily klines files into a single CSV file for a given time range.
Reads from data/raw_data/{symbol}/klines/{interval}/YYYY/MM/*.csv.zst
Saves to data/{symbol}_klines_{interval}_{start_date}_{end_date}.csv
"""
import sys
from pathlib import Path
import pandas as pd
import zstandard as zstd
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent


def load_kline_file(file_path: Path) -> pd.DataFrame:
    """Load a single kline .zst file."""
    try:
        dctx = zstd.ZstdDecompressor()
        with open(file_path, 'rb') as f_in:
            reader = dctx.stream_reader(f_in)
            df = pd.read_csv(reader)
        return df
    except Exception as e:
        print(f"  ⚠️  Error loading {file_path.name}: {e}")
        return pd.DataFrame()


def get_kline_files(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    base_dir: Path
) -> list:
    """Get list of kline files for the date range."""
    files = []
    current_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        year = current_dt.strftime("%Y")
        month = current_dt.strftime("%m")
        
        # Build file path
        file_path = base_dir / symbol / "klines" / interval / year / month / f"{symbol}-{interval}-{date_str}.csv.zst"
        
        if file_path.exists():
            files.append(file_path)
        
        current_dt += timedelta(days=1)
    
    return files


def merge_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Merge kline files into a single DataFrame.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '5m', '15m', '1h')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        output_dir: Output directory (default: data/)
    
    Returns:
        Merged DataFrame with all klines
    """
    # Default paths
    base_dir = project_root / "data" / "raw_data"
    if output_dir is None:
        output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Merging Klines: {symbol} | {interval} | {start_date} to {end_date}")
    print("=" * 70)
    
    # Get list of files
    print("\n📂 Scanning for kline files...")
    files = get_kline_files(symbol, interval, start_date, end_date, base_dir)
    
    if not files:
        print(f"⚠️  No kline files found for {symbol} {interval} in date range")
        return pd.DataFrame()
    
    print(f"✅ Found {len(files)} daily kline files")
    
    # Load and merge all files
    print(f"\n📊 Loading and merging klines...")
    all_data = []
    
    for file_path in tqdm(files, desc="Processing", unit="file", colour="green"):
        df = load_kline_file(file_path)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("⚠️  No valid data loaded")
        return pd.DataFrame()
    
    # Concatenate all data
    print("\n🔗 Concatenating data...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    print("🔄 Sorting by timestamp...")
    merged_df = merged_df.sort_values('open_time').reset_index(drop=True)
    
    # Remove duplicates (in case of overlap)
    print("🧹 Removing duplicates...")
    original_len = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['open_time'], keep='first')
    duplicates_removed = original_len - len(merged_df)
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate rows")
    
    # Save to CSV
    output_filename = f"{symbol}_klines_{interval}_{start_date}_{end_date}.csv"
    output_path = output_dir / output_filename
    
    print(f"\n💾 Saving to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ Merge Complete")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Columns: {', '.join(merged_df.columns)}")
    print(f"Time range: {merged_df['open_time'].min()} → {merged_df['open_time'].max()}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Show sample data
    print("\n📋 First 5 rows:")
    print(merged_df.head())
    print("\n📋 Last 5 rows:")
    print(merged_df.tail())
    
    return merged_df


def main():
    """Main execution."""
    # Configuration - Modify these parameters as needed
    symbol = "BTCUSDT"
    interval = "5m"  # 1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    
    try:
        merged_df = merge_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if merged_df.empty:
            print("\n⚠️  No data was merged")
            sys.exit(1)
        
        print("\n✅ Success!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
