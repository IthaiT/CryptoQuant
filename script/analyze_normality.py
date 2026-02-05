"""
Normality Analysis Script
Analyze return normality for any bar type (Dollar, Volume, Tick, Time bars)

Usage:
    python analyze_normality.py --file "data/preprocess_data/BTCUSDT_dollar_bars_2025.csv"
    python analyze_normality.py --file "data/BTCUSDT_klines_5m_2025-01-01_2025-12-31.csv"
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')

# Configure matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.facecolor'] = 'white'

project_root = Path(__file__).parent.parent


def load_data(file_path: Path) -> tuple[pd.DataFrame, str]:
    """Load CSV data and auto-detect bar type"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Auto-detect bar type from filename
    filename = file_path.stem.lower()
    if 'dollar' in filename:
        bar_type = "Dollar Bar"
    elif 'volume' in filename:
        bar_type = "Volume Bar"
    elif 'tick' in filename:
        bar_type = "Tick Bar"
    elif 'klines' in filename or any(t in filename for t in ['1m', '5m', '15m', '1h', '4h', '1d']):
        bar_type = "Time Bar"
    else:
        bar_type = "Unknown Bar"
    
    return df, bar_type


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate return metrics"""
    df = df.copy()
    df['returns'] = (df['close'] - df['open']) / df['open']
    df['log_returns'] = np.log(df['close'] / df['open'])
    df['abs_returns'] = np.abs(df['returns'])
    return df


def normality_tests(returns: np.ndarray) -> dict:
    """Perform multiple normality tests"""
    returns_clean = returns[~np.isnan(returns)]
    
    results = {}
    
    # 1. Shapiro-Wilk test
    if len(returns_clean) <= 5000:
        stat, p = stats.shapiro(returns_clean)
    else:
        stat, p = stats.shapiro(np.random.choice(returns_clean, 5000, replace=False))
    results['Shapiro-Wilk'] = {'stat': stat, 'p_value': p}
    
    # 2. Kolmogorov-Smirnov test
    stat, p = stats.kstest(returns_clean, 'norm', args=(returns_clean.mean(), returns_clean.std()))
    results['Kolmogorov-Smirnov'] = {'stat': stat, 'p_value': p}
    
    # 3. Jarque-Bera test
    stat, p = stats.jarque_bera(returns_clean)
    results['Jarque-Bera'] = {'stat': stat, 'p_value': p}
    
    # 4. Anderson-Darling test
    result = stats.anderson(returns_clean, dist='norm')
    results['Anderson-Darling'] = {
        'stat': result.statistic,
        'critical': result.critical_values.tolist(),
        'levels': result.significance_level.tolist()
    }
    
    return results


def generate_report(df: pd.DataFrame, bar_type: str, output_dir: Path):
    """Generate statistical analysis report"""
    returns = df['returns'].dropna()
    
    report_file = output_dir / "normality_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{bar_type} - Normality Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Data overview
        f.write("Data Overview:\n")
        f.write(f"  Total bars:      {len(df):,}\n")
        f.write(f"  Valid returns:   {len(returns):,}\n")
        if 'open_time' in df.columns:
            f.write(f"  Time range:      {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}\n")
        f.write("\n")
        
        # Descriptive statistics
        f.write("Descriptive Statistics:\n")
        f.write(f"  Mean:            {returns.mean():.8f}\n")
        f.write(f"  Median:          {returns.median():.8f}\n")
        f.write(f"  Std Dev:         {returns.std():.8f}\n")
        f.write(f"  Skewness:        {returns.skew():.6f}\n")
        f.write(f"  Kurtosis:        {returns.kurtosis():.6f}\n")
        f.write(f"  Min:             {returns.min():.8f}\n")
        f.write(f"  Max:             {returns.max():.8f}\n\n")
        
        # Normality test results
        f.write("Normality Tests (Significance Level: 0.05):\n")
        f.write("-" * 80 + "\n")
        
        test_results = normality_tests(returns.values)
        
        for test_name, result in test_results.items():
            f.write(f"\n{test_name}:\n")
            if test_name == 'Anderson-Darling':
                f.write(f"  Statistic:       {result['stat']:.6f}\n")
                f.write(f"  Critical values: {result['critical']}\n")
                f.write(f"  Significance:    {result['levels']}%\n")
            else:
                f.write(f"  Statistic:       {result['stat']:.6f}\n")
                f.write(f"  P-value:         {result['p_value']:.6f}\n")
                is_normal = "Yes" if result['p_value'] > 0.05 else "No"
                f.write(f"  Normal (p>0.05): {is_normal}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Interpretation:\n")
        f.write("  - P-value > 0.05: Data likely follows normal distribution\n")
        f.write("  - P-value < 0.05: Data significantly deviates from normality\n")
        f.write("  - Kurtosis > 3: Heavy tails (more extreme values)\n")
        f.write("  - Skewness ~ 0: Symmetric distribution\n")
        f.write("=" * 80 + "\n")
    
    print(f"OK: Report saved - {report_file.name}")


def plot_distributions(df: pd.DataFrame, bar_type: str, output_dir: Path):
    """Generate 6 distribution analysis plots"""
    returns = df['returns'].dropna().values
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{bar_type} - Returns Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram + Normal distribution
    ax = axes[0, 0]
    ax.hist(returns, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label='Normal Distribution')
    ax.set_xlabel('Returns', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Histogram vs Normal Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Q-Q plot
    ax = axes[0, 1]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Quantile-Quantile)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.get_lines()[0].set_markersize(4)
    
    # 3. Box plot
    ax = axes[0, 2]
    bp = ax.boxplot(returns, vert=True, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Returns', fontsize=11)
    ax.set_title('Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Log-scale histogram
    ax = axes[1, 0]
    ax.hist(returns, bins=100, density=True, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Returns', fontsize=11)
    ax.set_ylabel('Log Density', fontsize=11)
    ax.set_title('Log-scale Histogram (Heavy Tails)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, which='major')  # Only major gridlines for cleaner look
    
    # 5. Kernel Density Estimate
    ax = axes[1, 1]
    pd.Series(returns).plot(kind='kde', ax=ax, color='purple', linewidth=2.5)
    ax.set_xlabel('Returns', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Kernel Density Estimate (KDE)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Cumulative Distribution Function
    ax = axes[1, 2]
    ax.hist(returns, bins=100, density=True, cumulative=True, alpha=0.7, 
            color='coral', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Returns', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK: Saved - distributions.png")


def plot_timeseries(df: pd.DataFrame, bar_type: str, output_dir: Path):
    """Generate time series analysis plots"""
    # Sample for visualization (avoid too many points)
    sample_rate = max(1, len(df) // 10000)
    df_plot = df.iloc[::sample_rate].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'{bar_type} - Time Series Analysis (sample: every {sample_rate}th bar)', 
                 fontsize=16, fontweight='bold')
    
    x = range(len(df_plot))
    
    # 1. Price
    ax = axes[0]
    ax.plot(x, df_plot['close'].values, color='blue', linewidth=1, alpha=0.8)
    ax.fill_between(x, df_plot['open'].values, df_plot['close'].values,
                     where=(df_plot['close'] >= df_plot['open']),
                     alpha=0.2, color='green', label='Up')
    ax.fill_between(x, df_plot['open'].values, df_plot['close'].values,
                     where=(df_plot['close'] < df_plot['open']),
                     alpha=0.2, color='red', label='Down')
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('Close Price', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Returns
    ax = axes[1]
    colors = ['green' if ret > 0 else 'red' for ret in df_plot['returns'].values]
    ax.bar(x, df_plot['returns'].values, color=colors, alpha=0.6, width=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Returns', fontsize=11)
    ax.set_title('Returns Time Series', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Volume
    ax = axes[2]
    ax.bar(x, df_plot['volume'].values, color='purple', alpha=0.6, width=1)
    ax.set_ylabel('Volume', fontsize=11)
    ax.set_xlabel('Bar Index (sampled)', fontsize=11)
    ax.set_title('Volume', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK: Saved - timeseries.png")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze normality of bar data returns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python analyze_normality.py --file \"data/preprocess_data/BTCUSDT_dollar_bars_2025.csv\""
    )
    parser.add_argument('--file', type=str, required=True, help='Path to CSV bar data file')
    args = parser.parse_args()
    
    # Parse file path
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = project_root / file_path
    
    print("\n" + "=" * 80)
    print("Normality Analysis")
    print("=" * 80)
    print(f"\nFile: {file_path.name}")
    
    # Load data
    df, bar_type = load_data(file_path)
    print(f"Type: {bar_type}")
    print(f"Data: {len(df):,} bars\n")
    
    # Calculate returns
    df = calculate_returns(df)
    
    # Create output directory
    output_dir = project_root / "data" / "preprocess_data" / f"{file_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir.relative_to(project_root)}/\n")
    
    # Generate report
    print("Generating report...")
    generate_report(df, bar_type, output_dir)
    
    print("\nGenerating plots...")
    plot_distributions(df, bar_type, output_dir)
    plot_timeseries(df, bar_type, output_dir)
    
    print("\n" + "=" * 80)
    print(f"OK: Analysis complete! Results saved to: {output_dir.relative_to(project_root)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
