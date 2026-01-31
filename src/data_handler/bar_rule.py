"""
Bar Rule Module: Defines various bar generation rules using strategy pattern.

Supported types: DollarBar, VolumeBar, TickBar
"""
from typing import Dict, Any
from abc import ABC, abstractmethod


class BarRule(ABC):
    """Abstract base class for bar rules."""
    
    @abstractmethod
    def init_bar(self, ts: int, price: float, amount: float) -> Dict[str, Any]:
        """Initialize a new bar with the first trade."""
        pass

    @abstractmethod
    def update_bar(self, bar: Dict[str, Any], ts: int, price: float, amount: float) -> None:
        """Update existing bar with a new trade."""
        pass

    @abstractmethod
    def should_close(self, bar: Dict[str, Any]) -> bool:
        """Determine if the current bar should be closed."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset cumulative counters after bar closes."""
        pass


class DollarBarRule(BarRule):
    """Closes bar when cumulative dollar volume reaches threshold."""
    
    def __init__(self, threshold: float) -> None:
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
        return self.cumulative >= self.threshold
    
    def reset(self) -> None:
        self.cumulative = 0.0


class VolumeBarRule(BarRule):
    """Closes bar when cumulative volume reaches threshold."""
    
    def __init__(self, threshold: float) -> None:
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
    
    def reset(self) -> None:
        self.cumulative = 0.0


class TickBarRule(BarRule):
    """Closes bar after fixed number of trades."""
    
    def __init__(self, threshold: int) -> None:
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
    
    def reset(self) -> None:
        self.trade_count = 0
