from typing import Dict, Any
from enum import IntEnum
import pandas as pd

class Position(IntEnum):
    EXIT = 0
    LONG = 1
    SHORT = -1

class StrategyBase:
    """Base class for investment strategies."""

    def info(self) -> Dict[str, Any]:
        """Returns general informaiton about the strategy."""
        raise NotImplementedError

    def run(self, data: pd.DataFrame):
        """Run strategy on data."""
        raise NotImplementedError()