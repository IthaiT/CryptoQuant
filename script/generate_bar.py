"""
Unified Bar Generator - Supports Dollar, Volume, Tick Bars
Converts raw aggTrades data into various bar types with flexible time granularity.
Supports command-line arguments for easy configuration.
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_handler.bar_generator import BarGenerator
from src.data_handler.bar_rule import DollarBarRule, VolumeBarRule, TickBarRule


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate financial bars from raw trade data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dollar bars for entire range
  python generate_bar.py --bar-type dollar --threshold 10000000 --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
  
  # Generate daily dollar bars
  python generate_bar.py --bar-type dollar --threshold 10000000 --symbol BTCUSDT --start 2023-01-01 --end 2023-01-31 --period day
  
  # Generate monthly volume bars
  python generate_bar.py --bar-type volume --threshold 1000 --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31 --period month
  
  # Generate tick bars
  python generate_bar.py --bar-type tick --threshold 5000 --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
        """
    )
    
    parser.add_argument(
        "--bar-type",
        type=str,
        choices=["dollar", "volume", "tick"],
        default="dollar",
        help="Type of bar to generate (default: dollar)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold for bar closure (dollar amount, volume amount, or tick count)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: same as start)"
    )
    parser.add_argument(
        "--period",
        type=str,
        choices=["day", "month", "year", "none"],
        default="none",
        help="Period for grouping output (default: none - single file)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/preprocess_data)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    return parser.parse_args()


def get_bar_rule(bar_type: str, threshold: float):
    """Get appropriate bar rule instance."""
    if bar_type == "dollar":
        return DollarBarRule(threshold=threshold)
    elif bar_type == "volume":
        return VolumeBarRule(threshold=threshold)
    elif bar_type == "tick":
        return TickBarRule(threshold=int(threshold))
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")


def get_output_path(base_dir: Path, symbol: str, bar_type: str, threshold: float, 
                    period: str, period_key: str = None) -> Path:
    """
    Get organized output path based on bar type and period.
    
    Structure:
    - No period: {base_dir}/{bar_type}_bars/{symbol}/{symbol}_{bar_type}_bars_{threshold}_{date_range}.csv
    - With period: {base_dir}/{bar_type}_bars/{symbol}/{year}/{month?}/{symbol}_{bar_type}_bars_{period_key}.csv
    """
    bars_dir = base_dir / f"{bar_type}_bars" / symbol
    
    if period == "none" or period_key is None:
        bars_dir.mkdir(parents=True, exist_ok=True)
        return bars_dir
    
    # Organize by period hierarchy
    if period == "day":
        year, month, day = period_key.split('-')
        output_dir = bars_dir / year / month
    elif period == "month":
        year, month = period_key.split('-')
        output_dir = bars_dir / year
    else:  # year
        output_dir = bars_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_threshold(bar_type: str, threshold: float) -> str:
    """Format threshold for display and filename."""
    if bar_type == "dollar":
        if threshold >= 1_000_000:
            return f"{int(threshold/1_000_000)}m"
        elif threshold >= 1_000:
            return f"{int(threshold/1_000)}k"
        else:
            return str(int(threshold))
    elif bar_type == "volume":
        return str(int(threshold))
    else:  # tick
        return str(int(threshold))


def save_bars(df: pd.DataFrame, output_path: Path, symbol: str, bar_type: str, 
              threshold: float, period_key: str = None):
    """Save bars DataFrame to CSV with appropriate filename."""
    threshold_str = format_threshold(bar_type, threshold)
    
    if period_key:
        filename = f"{symbol}_{bar_type}_bars_{threshold_str}_{period_key}.csv"
    else:
        filename = f"{symbol}_{bar_type}_bars_{threshold_str}.csv"
    
    output_file = output_path / filename
    df.to_csv(output_file, index=False)
    return output_file


def print_statistics(stats_list: List[Dict[str, Any]], bar_type: str):
    """Print summary statistics for generated bars."""
    if not stats_list:
        return
    
    df_stats = pd.DataFrame(stats_list)
    
    print(f"\n{'='*70}")
    print(f"GENERATION SUMMARY - {bar_type.upper()} BARS")
    print(f"{'='*70}")
    print(f"{'Period':<15} {'Bars':>8} {'Avg Price':>12} {'Total Volume':>15} {'Total Trades':>15}")
    print("-" * 70)
    
    for _, row in df_stats.iterrows():
        period_str = str(row.get('period', 'all'))[:15]
        print(f"{period_str:<15} {int(row['num_bars']):>8} "
              f"${row['avg_price']:>11.2f} {row['total_volume']:>15.2f} {int(row['total_trades']):>15}")
    
    print("-" * 70)
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'Total periods:':<30} {len(df_stats)}")
    print(f"{'Total bars:':<30} {int(df_stats['num_bars'].sum())}")
    print(f"{'Avg bars per period:':<30} {df_stats['num_bars'].mean():.1f}")
    print(f"{'Total trades:':<30} {int(df_stats['total_trades'].sum())}")
    print(f"{'Average price:':<30} ${df_stats['avg_price'].mean():.2f}")
    print(f"{'Min price:':<30} ${df_stats['min_price'].min():.2f}")
    print(f"{'Max price:':<30} ${df_stats['max_price'].max():.2f}")
    print(f"{'='*70}\n")


def collect_stats(df: pd.DataFrame, period_key: str = None) -> Dict[str, Any]:
    """Collect statistics from a bars DataFrame."""
    return {
        'period': period_key or 'all',
        'num_bars': len(df),
        'avg_price': df['close'].mean(),
        'min_price': df['close'].min(),
        'max_price': df['close'].max(),
        'total_volume': df['volume'].sum(),
        'total_trades': df['num_trades'].sum(),
    }


def main():
    """Main execution."""
    args = parse_args()
    
    # Normalize period argument
    period = None if args.period == "none" else args.period
    
    # Set output directory
    output_base = args.output_dir or (project_root / "data" / "preprocess_data")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print(f"\n{'='*70}")
    print(f"BAR GENERATION")
    print(f"{'='*70}")
    print(f"Bar Type: {args.bar_type.upper()}")
    print(f"Symbol: {args.symbol}")
    print(f"Threshold: {format_threshold(args.bar_type, args.threshold)}")
    print(f"Period: {args.start} to {args.end or args.start}")
    print(f"Grouping: {period or 'single file'}")
    print(f"Output: {output_base}")
    print(f"{'='*70}\n")
    
    try:
        # Create bar rule and generator
        bar_rule = get_bar_rule(args.bar_type, args.threshold)
        bar_gen = BarGenerator()
        
        # Generate bars
        result = bar_gen.generate(
            bar_rule=bar_rule,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            period=period,
            show_progress=not args.no_progress
        )
        
        stats_list = []
        
        # Handle results based on period
        if period is None:
            # Single DataFrame
            if result.empty:
                print("⚠️  No bars generated!")
                return
            
            output_path = get_output_path(output_base, args.symbol, args.bar_type, 
                                         args.threshold, "none")
            output_file = save_bars(result, output_path, args.symbol, args.bar_type, 
                                   args.threshold)
            
            print(f"\n✅ Successfully saved {len(result)} bars to:")
            print(f"   {output_file}\n")
            
            stats_list.append(collect_stats(result))
        else:
            # Generator of (period_key, DataFrame) tuples
            for period_key, df_period in result:
                if df_period.empty:
                    continue
                
                output_path = get_output_path(output_base, args.symbol, args.bar_type, 
                                             args.threshold, period, period_key)
                output_file = save_bars(df_period, output_path, args.symbol, args.bar_type, 
                                       args.threshold, period_key)
                
                stats_list.append(collect_stats(df_period, period_key))
            
            if not stats_list:
                print("⚠️  No bars generated!")
                return
        
        # Print summary statistics
        print_statistics(stats_list, args.bar_type)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
