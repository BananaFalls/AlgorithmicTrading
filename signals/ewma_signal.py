"""
EWMA Signal Module - Following Robert Carver's systematic approach
Generates forecasts on a -20 to 20 scale (universal forecast standard)
"""

import pandas as pd
import numpy as np


class EWMASignal:
    """
    Exponential Weighted Moving Average Signal Generator

    Attributes:
        fast (int): Fast EMA span
        slow (int): Slow EMA span
    """

    def __init__(self, fast, slow):
        """
        Initialize EWMA signal

        Args:
            fast (int): Fast moving average span (e.g., 16)
            slow (int): Slow moving average span (e.g., 64)
        """
        self.fast = fast
        self.slow = slow
        self.name = f"EWMA_{fast}_{slow}"

    def get_forecast(self, prices, train_prices=None):
        """
        Generate forecast signal on -20 to 20 scale

        Args:
            prices (pd.Series): Price data to generate forecast for
            train_prices (pd.Series, optional): Training data for scaling factor.
                                               If None, uses prices for scaling.

        Returns:
            pd.Series: Forecast values between -20 and 20
        """
        # Use training prices for scaling if provided, else use test prices
        scaling_data = train_prices if train_prices is not None else prices

        # Step 1: Calculate EWMA
        fast_ewma = prices.ewm(span=self.fast, adjust=False).mean()
        slow_ewma = prices.ewm(span=self.slow, adjust=False).mean()

        # Step 2: Get raw signal
        raw_signal = (fast_ewma - slow_ewma) / prices

        # Step 3: Calculate scaling factor from training data
        scaling_data_ewma = scaling_data.ewm(span=self.fast, adjust=False).mean()
        scaling_data_slow = scaling_data.ewm(span=self.slow, adjust=False).mean()
        scaling_raw = (scaling_data_ewma - scaling_data_slow) / scaling_data
        scaling_factor = 10 / scaling_raw.abs().mean()

        # Step 4: Scale forecast to -20 to 20 range
        forecast = (raw_signal * scaling_factor).clip(-20, 20)

        return forecast

    def __repr__(self):
        return f"{self.name}"
