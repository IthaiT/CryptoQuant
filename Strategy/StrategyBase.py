from typing import Dict, Any
import pandas as pd

EXIT_POSITION = 0
LONG_POSITION = 1
SHORT_POSITION = -1

class StrategyBase:
    """Base class for investment strategies."""

    def info(self) -> Dict[str, Any]:
        """Returns general informaiton about the strategy."""
        raise NotImplementedError

    def run(self, data: pd.DataFrame):
        """Run strategy on data."""
        raise NotImplementedError()