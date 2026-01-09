from StrategyBase import *
import talib
import numpy as np
from typing import Dict, Any


class RSIStrategy(StrategyBase):
    """Strategy based on RSI."""

    NAME = "RSI"

    def __init__(self,
                 window_size: int = 14,
                 enter_long=None,
                 exit_long=None,
                 enter_short=None,
                 exit_short=None):
        self.window_size = window_size
        self.enter_long = enter_long
        self.exit_long = exit_long
        self.enter_short = enter_short
        self.exit_short = exit_short
        self.name = RSIStrategy.NAME
        # f"{RSIStrategy.NAME}(" +\
        #     "window={self.window_size}," +\
        #     "[{self.oversold}, {self.overbought}])"

    def info(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.name,
            'window_size': self.window_size,
            'enter_long': self.enter_long,
            'exit_long': self.exit_long,
            'enter_short': self.enter_short,
            'exit_short': self.exit_short
        }

    def run(self, data: pd.DataFrame):
        array = data['close_price'].to_numpy()

        rsi = talib.RSI(array, timeperiod=self.window_size)
        enter_long = rsi > (self.enter_long or np.infty)
        exit_long = rsi < (self.exit_long or -np.infty)
        enter_short = rsi < (
            self.enter_short or -np.infty)
        exit_short = rsi > (self.exit_short or np.infty)

        positions = np.full(rsi.shape, np.nan)
        positions[exit_long | exit_short] = EXIT_POSITION
        positions[enter_long] = LONG_POSITION
        positions[enter_short] = SHORT_POSITION

        # Fix the first position
        if np.isnan(positions[0]):
            positions[0] = EXIT_POSITION

        mask = np.isnan(positions)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        positions[mask] = positions[idx[mask]]

        return positions.astype(np.int32)
        # result = rsi_obos(rsi, self.oversold, self.overbought)

        # run_info = {
        #     'rsi': rsi
        # }

        # return result  # , run_info