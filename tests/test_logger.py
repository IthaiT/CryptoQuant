"""
Test script for QuantLogger
Run this to see the logger in action.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.logger import QuantLogger, LoggerPresets
import time

def demo_basic_logging():
    """Demonstrate basic logging features."""
    print("\n" + "=" * 80)
    print("📋 Basic Logging Demo")
    print("=" * 80 + "\n")
    
    # Get different loggers
    backtest = LoggerPresets.backtest()
    strategy = LoggerPresets.strategy()
    data = LoggerPresets.data_processing()
    risk = LoggerPresets.risk_management()
    
    # Test all log levels
    backtest.debug('Analyzing historical data from 2020-2025')
    backtest.info('Backtest started: BTCUSDT, Period: 2023-01-01 to 2023-12-31')
    
    strategy.info('Signal generated: LONG position at $42,500')
    strategy.info('Entry confirmed with RSI(14)=35.2')
    
    data.warning('Data quality check: 3 missing values detected in volume column')
    
    risk.error('Position size exceeds maximum risk limit (5%)')
    risk.critical('Emergency stop: Account drawdown reached 20%')


def demo_timer():
    """Demonstrate timing functionality."""
    print("\n" + "=" * 80)
    print("⏱️  Timer Demo")
    print("=" * 80 + "\n")
    
    logger = QuantLogger.get_logger('performance')
    
    # Timer for data loading
    with QuantLogger.timer('Load 1M bars', 'performance'):
        time.sleep(0.8)
    
    # Timer for calculation
    with QuantLogger.timer('Calculate indicators', 'performance'):
        time.sleep(0.5)
    
    # Timer for backtest
    with QuantLogger.timer('Run backtest', 'performance'):
        time.sleep(1.2)


def demo_real_scenario():
    """Demonstrate a realistic trading scenario."""
    print("\n" + "=" * 80)
    print("🎯 Realistic Trading Scenario")
    print("=" * 80 + "\n")
    
    data_log = LoggerPresets.data_processing()
    strategy_log = LoggerPresets.strategy()
    risk_log = LoggerPresets.risk_management()
    
    # Data loading phase
    data_log.info('Loading BTCUSDT dollar bars from 2023-01-01')
    with QuantLogger.timer('Data loading', 'data'):
        time.sleep(0.3)
    data_log.info('Loaded 156,234 bars successfully')
    
    # Strategy analysis
    strategy_log.info('Initializing RSI strategy (period=14, oversold=30)')
    strategy_log.info('Scanning for entry signals...')
    time.sleep(0.2)
    
    strategy_log.info('Signal detected at 2023-03-15 14:30:00')
    strategy_log.info('  Price: $28,450.00')
    strategy_log.info('  RSI: 28.5 (oversold)')
    strategy_log.info('  Signal: LONG')
    
    # Risk check
    risk_log.info('Position size calculation:')
    risk_log.info('  Account balance: $100,000')
    risk_log.info('  Risk per trade: 2%')
    risk_log.info('  Stop loss: $27,950 (-1.76%)')
    risk_log.info('  Position size: 0.7 BTC')
    risk_log.info('✅ Risk check passed')
    
    # Execution
    strategy_log.info('Order submitted: BUY 0.7 BTC @ $28,450')
    time.sleep(0.1)
    strategy_log.info('✅ Order filled: Avg price $28,452.50')
    
    # Position management
    time.sleep(0.3)
    strategy_log.info('Price update: $29,100 (+2.27%)')
    strategy_log.info('Trailing stop updated to $28,700')
    
    time.sleep(0.2)
    strategy_log.warning('Price dropped to $28,650, approaching stop loss')
    
    time.sleep(0.1)
    strategy_log.info('Stop loss triggered at $28,700')
    strategy_log.info('Trade closed: PnL = +$172.50 (+0.17%)')


def demo_error_handling():
    """Demonstrate error logging."""
    print("\n" + "=" * 80)
    print("⚠️  Error Handling Demo")
    print("=" * 80 + "\n")
    
    data_log = LoggerPresets.data_processing()
    
    try:
        data_log.info('Attempting to load data from invalid path')
        with QuantLogger.timer('Load data', 'data'):
            # Simulate an error
            raise FileNotFoundError('Data file not found: /path/to/data.csv')
    except FileNotFoundError as e:
        data_log.error(f'Failed to load data: {e}')
    
    data_log.info('Falling back to cached data')


if __name__ == '__main__':
    # Setup logger with custom configuration
    QuantLogger.setup(
        console_level='DEBUG',
        file_level='DEBUG'
    )
    
    print("🚀 QuantLogger Test Suite")
    print("=" * 80)
    
    # Run demos
    demo_basic_logging()
    demo_timer()
    demo_real_scenario()
    demo_error_handling()
    
    print("\n" + "=" * 80)
    print("✅ All demos completed!")
    print("=" * 80)
    print(f"\n📁 Log files saved to: {QuantLogger._root_log_dir}")
