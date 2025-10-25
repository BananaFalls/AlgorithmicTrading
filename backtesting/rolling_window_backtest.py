import pandas as pd
import numpy as np


class RollingWindowBacktest:
    """
    Walk-forward backtest with rolling train/test windows

    This implements Carver's preferred validation method:
    - Train on historical data
    - Test on out-of-sample data
    - Roll forward and repeat

    Attributes:
        signal: Signal generator object (must have get_forecast method)
        sizing: Position sizing object (must have get_position_size method)
        train_window (int): Training period in days (default 252 = 1 year)
        test_window (int): Testing period in days (default 63 = 3 months)
    """

    def __init__(self, signal, sizing, train_window=252, test_window=63):
        """
        Initialize rolling window backtest

        Args:
            signal: Signal generator with get_forecast(prices, train_prices) method
            sizing: Position sizing with get_position_size(forecast, prices, capital) method
            train_window (int): Days for training period
            test_window (int): Days for testing period
        """
        self.signal = signal
        self.sizing = sizing
        self.train_window = train_window
        self.test_window = test_window

    def run(self, prices, capital=100000, transaction_cost=0.0001):
        """
        Run walk-forward backtest

        Args:
            prices (pd.Series): Daily price data with datetime index
            capital (float): Initial capital
            transaction_cost (float): Transaction cost as decimal (default 0.0001 = 1 bps)

        Returns:
            dict: Contains 'returns', 'forecasts', 'positions', 'cumulative_returns'
        """
        all_forecasts = []
        all_positions = []
        all_returns = []
        all_costs = []
        timestamps = []

        # Walk forward through data
        for i in range(self.train_window, len(prices) - self.test_window, self.test_window):
            # Split into train and test
            train_prices = prices.iloc[i - self.train_window : i]
            test_prices = prices.iloc[i : i + self.test_window]

            # Get signal from training data
            forecast = self.signal.get_forecast(test_prices, train_data=train_prices)

            # Size positions based on forecast
            positions = self.sizing.get_position_size(forecast, test_prices, capital)

            # Calculate returns
            test_returns = test_prices.pct_change()
            strategy_returns = positions.shift(1) * test_returns

            # Calculate transaction costs
            position_changes = positions.diff().abs()
            costs = position_changes * transaction_cost # costs proportional to size of bets

            # Net returns after costs
            net_returns = strategy_returns - costs

            # Store results
            valid_idx = net_returns.dropna().index
            all_forecasts.extend(forecast.loc[valid_idx].values)
            all_positions.extend(positions.loc[valid_idx].values)
            all_returns.extend(net_returns.loc[valid_idx].values)
            all_costs.extend(costs.loc[valid_idx].values)
            timestamps.extend(valid_idx)

        # Create result series with datetime index
        result_index = pd.DatetimeIndex(timestamps)

        results = {
            "returns": pd.Series(all_returns, index=result_index),
            "forecasts": pd.Series(all_forecasts, index=result_index),
            "positions": pd.Series(all_positions, index=result_index),
            "costs": pd.Series(all_costs, index=result_index),
        }

        # Calculate cumulative returns
        results["cumulative_returns"] = (1 + results["returns"]).cumprod()

        return results

    def calculate_metrics(self, results):
        """
        Calculate performance metrics

        Args:
            results (dict): Output from run() method

        Returns:
            dict: Performance metrics
        """
        returns = results["returns"]

        # Avoid division by zero
        if returns.std() == 0 or len(returns) == 0:
            return None

        # Calculate metrics
        sharpe = returns.mean() / returns.std() * np.sqrt(256)
        total_return = (1 + returns).cumprod().iloc[-1] - 1
        annual_return = returns.mean() * 256

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = (
            (returns[returns > 0].sum() / returns[returns < 0].abs().sum())
            if (returns < 0).any()
            else np.inf
        )

        # Transaction costs
        total_costs = results["costs"].sum()

        metrics = {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "std_dev": returns.std(),
            "total_costs": total_costs,
            "num_trades": len(returns),
        }

        return metrics

    def __repr__(self):
        return f"RollingWindowBacktest(train={self.train_window}d, test={self.test_window}d)"
