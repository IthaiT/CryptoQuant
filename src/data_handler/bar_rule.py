"""
Bar Rule Module: Defines various bar generation rules using strategy pattern.

Supported types: DollarBar, VolumeBar, TickBar
"""
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BarRule(ABC):
    """Abstract base class for bar rules."""
    
    @abstractmethod
    def init_bar(self, ts: int, price: float, amount: float, previous_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        '''
        我希望我生成bar的频率是5-10min/根，对应一天就是144-288根bar，取中间216
        self.update_interval = 54 = 216/4，也就是期望6个小时更新一次threshold
        self.k = 14，期望回看的窗口是4.5day
        '''
        self.ema_threshold = threshold  # Current EMA-smoothed threshold
        self.k = 14  # EMA smoothing parameter
        self.alpha = 2 / (self.k + 1)  # EMA smoothing factor (2/(k+1))
        self.update_interval = 54  # Update threshold every 54 bars
        
    def _get_dynamic_threshold(self, previous_bars: List[Dict[str, Any]]) -> float:
        # Get the last 54 bars for calculation
        recent_bars = previous_bars[-self.update_interval:]
        
        # Step 1: Calculate average dollar volume over last 54 bars
        # This represents the current market activity level
        total_dollar_volume = sum(bar['dollar_volume'] for bar in recent_bars)
        recent_threshold = total_dollar_volume / self.update_interval
        
        # Step 2: Apply EMA smoothing to smooth out volatility
        # EMA formula: EMA_new = (recent_value × α) + (EMA_old × (1 - α))
        # where α = 2/(k+1), k=14 in this case
        self.ema_threshold = (recent_threshold * self.alpha) + (self.ema_threshold * (1 - self.alpha))
        
        return self.ema_threshold
    
    def init_bar(self, ts: int, price: float, amount: float, previous_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        value = price * amount
        self.cumulative = value
        if(len(previous_bars) % self.update_interval == 0):
            self.threshold = self._get_dynamic_threshold(previous_bars)
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

    def init_bar(self, ts: int, price: float, amount: float, previous_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def init_bar(self, ts: int, price: float, amount: float, previous_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
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
