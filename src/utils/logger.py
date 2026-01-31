"""
Quantitative Trading Logger System
Provides colorful, structured logging with performance monitoring capabilities.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
import time


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output (modifies the output only)."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Emoji indicators (for console only)
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '📊',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥',
    }
    
    def format(self, record):
        # Save original levelname
        original_levelname = record.levelname
        
        # Add color and icon to level name for display
        if original_levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[original_levelname]}{self.BOLD}{self.ICONS[original_levelname]} "
                f"{original_levelname}{self.RESET}"
            )
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname (important for file handlers!)
        record.levelname = original_levelname
        
        return result


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors, no emoji)."""
    pass


class QuantLogger:
    """
    Professional Logger for Quantitative Trading Systems.
    
    Features:
    - Colorful console output with emoji indicators
    - File logging with rotation
    - Performance timing utilities
    - Context-aware formatting (backtest/live/data)
    - Thread-safe operations
    
    Usage:
        >>> logger = QuantLogger.get_logger('strategy')
        >>> logger.info('Strategy initialized')
        >>> 
        >>> # With timer
        >>> with QuantLogger.timer('data_loading'):
        ...     load_data()
    """
    
    _loggers = {}  # Cache for logger instances
    _root_log_dir: Optional[Path] = None
    
    @classmethod
    def setup(
        cls,
        log_dir: Optional[Path] = None,
        console_level: str = 'INFO',
        file_level: str = 'DEBUG',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Setup global logging configuration.
        
        Args:
            log_dir: Directory for log files (default: project_root/logs)
            console_level: Console output level
            file_level: File output level
            max_bytes: Max size per log file before rotation
            backup_count: Number of backup files to keep
        """
        # Set default log directory
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / 'logs'
        
        cls._root_log_dir = Path(log_dir)
        cls._root_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configuration
        cls._console_level = getattr(logging, console_level.upper())
        cls._file_level = getattr(logging, file_level.upper())
        cls._max_bytes = max_bytes
        cls._backup_count = backup_count
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        context: Optional[str] = None,
        save_to_file: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (e.g., 'strategy', 'backtest', 'data_loader')
            context: Optional context label (e.g., 'LIVE', 'BACKTEST')
            save_to_file: Whether to save logs to file
        
        Returns:
            Configured logger instance
        """
        # Initialize if not done
        if cls._root_log_dir is None:
            cls.setup()
        
        # Create unique key for caching
        cache_key = f"{name}_{context or 'default'}"
        
        if cache_key in cls._loggers:
            return cls._loggers[cache_key]
        
        # Create new logger
        logger = logging.getLogger(cache_key)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # Clear existing handlers
        logger.propagate = False  # Prevent duplicate logs
        
        # Console handler with colors (colored output for terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(cls._console_level)
        
        # Format with context if provided
        console_format = cls._build_format(name, context, colored=True)
        console_handler.setFormatter(ColoredFormatter(console_format))
        logger.addHandler(console_handler)
        
        # File handler with rotation (plain format, no colors/emoji)
        if save_to_file:
            log_file = cls._root_log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=cls._max_bytes,
                backupCount=cls._backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(cls._file_level)
            
            # Plain format for file (no colors, no emoji)
            file_format = cls._build_format(name, context, colored=False)
            file_handler.setFormatter(PlainFormatter(file_format))
            logger.addHandler(file_handler)
        
        # Cache the logger
        cls._loggers[cache_key] = logger
        
        return logger
    
    @staticmethod
    def _build_format(name: str, context: Optional[str], colored: bool) -> str:
        """Build log format string."""
        if context:
            prefix = f"[{context}] "
        else:
            prefix = ""
        
        if colored:
            # Colored format (levelname already has color from ColoredFormatter)
            return f"%(asctime)s {prefix}%(levelname)s %(message)s"
        else:
            # Plain format for file
            return f"%(asctime)s {prefix}[%(levelname)s] [%(name)s] %(message)s"
    
    @classmethod
    def timer(cls, operation: str, logger_name: str = 'timer'):
        """
        Context manager for timing operations.
        
        Usage:
            >>> with QuantLogger.timer('data_loading'):
            ...     load_large_dataset()
            📊 INFO [data_loading] Started
            📊 INFO [data_loading] Completed in 2.34s
        """
        return _TimerContext(operation, cls.get_logger(logger_name))
    
    @classmethod
    def reset(cls):
        """Reset all loggers (useful for testing)."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


class _TimerContext:
    """Internal context manager for timing operations."""
    
    def __init__(self, operation: str, logger: logging.Logger):
        self.operation = operation
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"[{self.operation}] Started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"[{self.operation}] Completed in {elapsed:.2f}s")
        else:
            self.logger.error(
                f"[{self.operation}] Failed after {elapsed:.2f}s: "
                f"{exc_type.__name__}: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


# Convenience functions for quick usage
def get_logger(name: str, context: Optional[str] = None) -> logging.Logger:
    """Quick access to logger."""
    return QuantLogger.get_logger(name, context)


def setup_logging(**kwargs):
    """Quick access to setup."""
    QuantLogger.setup(**kwargs)


# Example usage and presets
class LoggerPresets:
    """Predefined logger configurations for common scenarios."""
    
    @staticmethod
    def backtest() -> logging.Logger:
        """Logger for backtesting."""
        return QuantLogger.get_logger('backtest', context='BACKTEST')
    
    @staticmethod
    def live_trading() -> logging.Logger:
        """Logger for live trading."""
        return QuantLogger.get_logger('trading', context='LIVE')
    
    @staticmethod
    def data_processing() -> logging.Logger:
        """Logger for data processing."""
        return QuantLogger.get_logger('data', context='DATA')
    
    @staticmethod
    def strategy() -> logging.Logger:
        """Logger for strategy logic."""
        return QuantLogger.get_logger('strategy', context='STRATEGY')
    
    @staticmethod
    def risk_management() -> logging.Logger:
        """Logger for risk management."""
        return QuantLogger.get_logger('risk', context='RISK')


if __name__ == '__main__':
    # Demo
    print("=" * 80)
    print("QuantLogger Demo")
    print("=" * 80)
    
    # Setup
    QuantLogger.setup(console_level='DEBUG')
    
    # Different loggers
    backtest_log = LoggerPresets.backtest()
    strategy_log = LoggerPresets.strategy()
    data_log = LoggerPresets.data_processing()
    
    # Test different levels
    backtest_log.debug('This is a debug message')
    backtest_log.info('Backtest initialized with 1000 bars')
    strategy_log.info('Strategy signal: BUY at $50,000')
    data_log.warning('Missing data for 2023-01-15')
    strategy_log.error('Stop loss triggered!')
    
    # Timer demo
    print("\n" + "=" * 80)
    print("Timer Demo")
    print("=" * 80)
    
    with QuantLogger.timer('data_loading', 'data'):
        time.sleep(1.5)  # Simulate work
    
    print("\n✅ Demo completed!")
