"""
Position Sizing Module - Following Robert Carver's approach
Converts forecasts (0-20 scale) to position sizes based on volatility targeting
"""

import pandas as pd
import numpy as np


class VolTargetSizing:
    """
    Volatility-Targeted Position Sizing

    Implements Carver's position sizing rule:
    Position = (Vol Target * Capital * Forecast/10) / (Asset Volatility * Price)

    Attributes:
        vol_target (float): Target volatility as decimal (e.g., 0.1 = 10%)
        lookback (int): Lookback period for volatility calculation (default 25 days)
    """

    def __init__(self, vol_target=0.25, lookback=25):
        """
        Initialize volatility target sizing

        Args:
            vol_target (float): Target annual volatility (default 0.25 = 25%)
            lookback (int): Lookback period for volatility (default 25 days)
        """
        self.vol_target = vol_target
        self.lookback = lookback

    def calculate_volatility(self, prices):
        """
        Calculate annualized volatility

        Args:
            prices (pd.Series): Price data

        Returns:
            pd.Series: Annualized volatility
        """
        returns = prices.pct_change()
        volatility = returns.ewm(span=self.lookback, adjust=False).std() * np.sqrt(256)
        return volatility

    def get_position_size(self, forecast, prices, capital):
        """
        Convert forecast to position size in dollars

        Args:
            forecast (pd.Series): Forecast values (-20 to 20 scale)
            prices (pd.Series): Asset prices
            capital (float): Total capital/portfolio value

        Returns:
            pd.Series: Position sizes in dollars (can be negative for shorts)
        """
        # Calculate volatility
        volatility = self.calculate_volatility(prices)

        # Instrument currency volatility (dollar terms)
        instrument_currency_vol = volatility * prices

        # Desired notional position
        notional_position = (self.vol_target * capital) / instrument_currency_vol

        # Scale by forecast (forecast is -20 to 20, divide by 10 to get -2 to 2 multiplier)
        positions = notional_position * (forecast / 10)

        return positions

    def get_leverage(self, position_size, prices, capital):
        """
        Calculate leverage ratio

        Args:
            position_size (pd.Series): Position sizes
            prices (pd.Series): Asset prices
            capital (float): Total capital

        Returns:
            pd.Series: Leverage (notional exposure / capital)
        """
        notional_exposure = (position_size * prices).abs()
        leverage = notional_exposure / capital
        return leverage

    def __repr__(self):
        return f"VolTargetSizing(vol_target={self.vol_target:.1%}, lookback={self.lookback})"
