"""
Portfolio Optimizer Module - Following Robert Carver's FDM approach
Builds portfolios using diversification multiplier (FDM) from forecast correlations
"""

import pandas as pd
import numpy as np


class PortfolioOptimizer:
    """
    Portfolio construction using Carver's Forecast Diversification Multiplier (FDM)

    Key principle: Use correlations between forecasts to build diversified portfolios
    FDM = 1 / sqrt(correlation of forecasts)

    Attributes:
        min_correlation (float): Minimum acceptable correlation for inclusion
        leverage (float): Maximum leverage to apply
    """

    def __init__(self, min_correlation=0.5, leverage=1.0):
        """
        Initialize portfolio optimizer

        Args:
            min_correlation (float): Minimum correlation to include signal (0.0 to 1.0)
            leverage (float): Maximum leverage to apply (default 1.0 = no leverage)
        """
        self.min_correlation = min_correlation
        self.leverage = leverage

    def calculate_correlations(self, forecasts_dict):
        """
        Calculate correlations between forecasts

        Args:
            forecasts_dict (dict): Dictionary of {signal_name: forecast_series}

        Returns:
            pd.DataFrame: Correlation matrix
        """
        forecasts_df = pd.DataFrame(forecasts_dict)
        correlations = forecasts_df.corr()
        return correlations

    def calculate_fdm(self, correlation_matrix):
        """
        Calculate Forecast Diversification Multiplier

        FDM = 1 / sqrt(average correlation)

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix of forecasts

        Returns:
            float: Diversification multiplier
        """
        # Get average correlation (excluding diagonal)
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        avg_correlation = correlation_matrix.values[mask].mean()

        # Avoid division by zero or negative correlations
        avg_correlation = max(avg_correlation, 0.0)

        fdm = 1 / np.sqrt(1 - avg_correlation) if avg_correlation < 1 else 1.0

        return fdm

    def combine_forecasts(self, forecasts_dict, correlation_weights=None):
        """
        Combine multiple forecasts into single portfolio forecast

        Args:
            forecasts_dict (dict): Dictionary of {signal_name: forecast_series}
            correlation_weights (dict, optional): Custom weights. If None, equal weight.

        Returns:
            pd.Series: Combined forecast
        """
        forecasts_df = pd.DataFrame(forecasts_dict)

        if correlation_weights is None:
            # Equal weight
            combined = forecasts_df.mean(axis=1)
        else:
            # Weighted by provided weights
            weights = [correlation_weights.get(col, 1.0) for col in forecasts_df.columns]
            weights = np.array(weights) / sum(weights)  # Normalize
            combined = (forecasts_df * weights).sum(axis=1)

        # Clip to -20 to 20 range
        combined = combined.clip(-20, 20)

        return combined

    def print_summary(self, correlation_matrix):
        """
        Print portfolio analysis summary

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
        """
        fdm = self.calculate_fdm(correlation_matrix)

        print("\n" + "=" * 60)
        print("FORECAST CORRELATION MATRIX")
        print("=" * 60)
        print(correlation_matrix.round(3))

        print("\n" + "=" * 60)
        print("PORTFOLIO ANALYSIS")
        print("=" * 60)
        print(f"Average Correlation: {correlation_matrix.values[~np.eye(len(correlation_matrix), dtype=bool)].mean():.3f}")
        print(f"Forecast Diversification Multiplier (FDM): {fdm:.3f}")
        print(f"  → Forecasts are {fdm:.1f}x more powerful when combined")
        print(f"  → FDM of 1.0 = forecasts uncorrelated")
        print(f"  → FDM of 2.0+ = high diversification benefit")

    def __repr__(self):
        return f"PortfolioOptimizer(min_corr={self.min_correlation:.2f}, leverage={self.leverage:.1f})"
