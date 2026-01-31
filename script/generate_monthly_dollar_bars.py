"""
Generate Monthly Dollar Bars from Raw Trade Data (2020-2025)
Converts aggTrades data into Dollar Bars, organized by year/month.
Generates statistical summary of all bars.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.bar_generator import DollarBar


class MonthlyDollarBarGenerator:
    """Generate and manage monthly dollar bars with statistics."""
    
    def __init__(self, symbol: str, threshold: float, base_output_dir: Path):
        """
        Initialize generator.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            threshold: Dollar threshold per bar
            base_output_dir: Base directory for output (will create subdirs)
        """
        self.symbol = symbol
        self.threshold = threshold
        self.base_output_dir = base_output_dir
        self.bar_generator = DollarBar(threshold=threshold)
        
        # Create base directory structure
        self.bars_dir = base_output_dir / "dollar_bars" / symbol
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics storage
        self.monthly_stats = []
    
    def generate_all_monthly_bars(self, start_date: str, end_date: str):
        """
        Generate dollar bars for each month in date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Generate all year-month combinations
        months_to_process = []
        current_dt = start_dt.replace(day=1)
        while current_dt <= end_dt:
            months_to_process.append(current_dt.strftime("%Y-%m"))
            # Move to next month
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        print(f"\n{'='*70}")
        print(f"Generating Monthly Dollar Bars: {self.symbol}")
        print(f"{'='*70}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Threshold: ${self.threshold:,.0f} per bar")
        print(f"Total months: {len(months_to_process)}")
        print(f"Output directory: {self.bars_dir}")
        print(f"{'='*70}\n")
        
        for i, year_month in enumerate(months_to_process, 1):
            year, month = year_month.split('-')
            month_start = f"{year_month}-01"
            
            # Calculate month end
            month_dt = datetime.fromisoformat(month_start)
            if month_dt.month == 12:
                next_month_dt = month_dt.replace(year=month_dt.year + 1, month=1)
            else:
                next_month_dt = month_dt.replace(month=month_dt.month + 1)
            month_end = (next_month_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            
            print(f"[{i}/{len(months_to_process)}] Processing {year_month}...", end=" ")
            
            try:
                # Generate bars for this month
                df_bars = self.bar_generator.generate(
                    symbol=self.symbol,
                    start_date=month_start,
                    end_date=month_end,
                )
                
                if df_bars.empty:
                    print("⊘ No data")
                    continue
                
                # Save to year/month subdirectory
                year_dir = self.bars_dir / year
                year_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = year_dir / f"{self.symbol}_dollar_bars_{year_month}.csv"
                df_bars.to_csv(output_file, index=False)
                
                # Collect statistics
                self._collect_stats(df_bars, year_month, output_file)
                
                print(f"✅ {len(df_bars)} bars saved")
            
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n{'='*70}")
        print(f"✅ Monthly bar generation complete!")
        print(f"{'='*70}\n")
    
    def _collect_stats(self, df_bars: pd.DataFrame, year_month: str, output_file: Path):
        """Collect statistics for a month's bars."""
        stats = {
            'year_month': year_month,
            'output_file': str(output_file.relative_to(self.base_output_dir)),
            'num_bars': len(df_bars),
            'start_timestamp': df_bars['timestamp'].min(),
            'end_timestamp': df_bars['timestamp'].max(),
            'avg_price': df_bars['close'].mean(),
            'min_price': df_bars['close'].min(),
            'max_price': df_bars['close'].max(),
            'avg_volume': df_bars['volume'].mean(),
            'total_volume': df_bars['volume'].sum(),
            'avg_trades_per_bar': df_bars['num_trades'].mean(),
            'total_trades': df_bars['num_trades'].sum(),
            'avg_dollar_volume': df_bars['dollar_volume'].mean(),
            'total_dollar_volume': df_bars['dollar_volume'].sum(),
        }
        self.monthly_stats.append(stats)
    
    def generate_summary_report(self):
        """Generate and save monthly summary statistics."""
        if not self.monthly_stats:
            print("⚠️  No statistics to report")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.monthly_stats)
        
        # Save summary report
        report_file = self.base_output_dir / f"dollar_bars_summary_{self.symbol}.csv"
        summary_df.to_csv(report_file, index=False)
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"MONTHLY DOLLAR BAR STATISTICS - {self.symbol}")
        print(f"{'='*70}\n")
        
        print(f"{'Year-Month':<15} {'Bars':>8} {'Avg Price':>12} {'Total Volume':>15} {'Total Trades':>15}")
        print("-" * 70)
        
        for _, row in summary_df.iterrows():
            print(f"{row['year_month']:<15} {int(row['num_bars']):>8} "
                  f"${row['avg_price']:>11.2f} {row['total_volume']:>15.2f} {int(row['total_trades']):>15}")
        
        print("-" * 70)
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'Total months processed:':<30} {len(summary_df)}")
        print(f"{'Total bars:':<30} {int(summary_df['num_bars'].sum())}")
        print(f"{'Avg bars per month:':<30} {summary_df['num_bars'].mean():.1f}")
        print(f"{'Total trades:':<30} {int(summary_df['total_trades'].sum())}")
        print(f"{'Avg trades per bar:':<30} {summary_df['avg_trades_per_bar'].mean():.1f}")
        print(f"{'Average price (overall):':<30} ${summary_df['avg_price'].mean():.2f}")
        print(f"{'Min price (across all):':<30} ${summary_df['min_price'].min():.2f}")
        print(f"{'Max price (across all):':<30} ${summary_df['max_price'].max():.2f}")
        print(f"{'Total dollar volume:':<30} ${summary_df['total_dollar_volume'].sum():,.0f}")
        
        print(f"\n✅ Summary report saved to:")
        print(f"   {report_file}\n")
        print(f"{'='*70}\n")


def main():
    """Main execution."""
    
    # Configuration
    symbol = "BTCUSDT"
    start_date = "2020-01-01"
    end_date = "2025-12-31"
    dollar_threshold = 10_000_000.0  # $10M per bar
    
    # Output directory
    output_dir = project_root / "data" / "preprocess_data"
    
    try:
        # Create generator
        generator = MonthlyDollarBarGenerator(
            symbol=symbol,
            threshold=dollar_threshold,
            base_output_dir=output_dir
        )
        
        # Generate all monthly bars
        generator.generate_all_monthly_bars(start_date, end_date)
        
        # Generate summary report
        generator.generate_summary_report()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
