"""
Main Backtesting Script - Modular Strategy Following Robert Carver's Approach
This script demonstrates:
1. Creating multiple signal generators (EWMA variations)
2. Position sizing based on volatility targeting
3. Walk-forward validation to avoid look-ahead bias
4. Portfolio construction using correlation analysis
5. Performance metrics and reporting
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from signals.ewma_signal import EWMASignal
from position_sizing.vol_target_sizing import VolTargetSizing
from backtesting.rolling_window_backtest import RollingWindowBacktest
# from portfolio_u.portfolio_optimizer import PortfolioOptimizer


def load_data(filepath):
    """Load crypto data from CSV"""
    df = pd.read_csv(
        filepath,
        index_col='timestamp',
        parse_dates=True
    )
    return df['close']


def run_single_strategy(prices, fast, slow, vol_target, capital):
    """
    Run a single EWMA strategy with rolling window backtest

    Args:
        prices (pd.Series): Price data
        fast (int): Fast EMA span
        slow (int): Slow EMA span
        vol_target (float): Volatility target
        capital (float): Initial capital

    Returns:
        dict: Results containing returns, forecasts, positions, metrics
    """
    # Create signal generator
    signal = EWMASignal(fast=fast, slow=slow)

    # Create position sizing
    sizing = VolTargetSizing(vol_target=vol_target)

    # Create backtest
    backtest = RollingWindowBacktest(
        signal=signal,
        sizing=sizing,
        train_window=252,  # 1 year training
        test_window=63     # 3 months testing
    )

    # Run backtest
    results = backtest.run(prices, capital=capital, transaction_cost=0.0001)

    # Calculate metrics
    metrics = backtest.calculate_metrics(results)

    return {
        'signal': signal,
        'results': results,
        'metrics': metrics,
        'backtest': backtest
    }


def main():
    """Main execution"""

    print("=" * 70)
    print("MODULAR STRATEGY BACKTEST - ROBERT CARVER'S SYSTEMATIC APPROACH")
    print("=" * 70)

    # Configuration
    DATA_PATH = r'C:\Users\diste\OneDrive\Desktop\Data\Crypto\BTC_USDT_1d.csv'
    VOL_TARGET = 0.25  # 25% volatility target
    CAPITAL = 100000   # $100k

    # Load data
    print("\n[1/3] Loading data...")
    prices = load_data(DATA_PATH)
    print(f"  ✓ Loaded {len(prices)} days of price data")
    print(f"  ✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Define EWMA variations
    ewma_variations = [
        # (4, 16),    # Very fast
        (8, 32),    # Very fast
        (16, 64),   # Fast
        # (24, 96),   # Fast
        (32, 128),  # Medium
        # (48, 192),  # Medium
    ]

    # Test all strategies
    print("\n[2/3] Running rolling window backtests...")
    print("-" * 70)

    all_results = {}
    all_forecasts = {}

    for fast, slow in ewma_variations:
        label = f"EWMA_{fast}_{slow}"
        print(f"\n  Testing {label}...")

        result = run_single_strategy(prices, fast, slow, VOL_TARGET, CAPITAL)

        all_results[label] = result
        all_forecasts[label] = result['results']['forecasts']

        # Print metrics
        metrics = result['metrics']
        if metrics:
            print(f"    Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
            print(f"    Total Return:     {metrics['total_return']:>8.1%}")
            print(f"    Annual Return:    {metrics['annual_return']:>8.1%}")
            print(f"    Max Drawdown:     {metrics['max_drawdown']:>8.1%}")
            print(f"    Win Rate:         {metrics['win_rate']:>8.1%}")
            print(f"    Profit Factor:    {metrics['profit_factor']:>8.2f}")

    # # Portfolio analysis
    # print("\n[3/3] Portfolio analysis (Carver's FDM approach)...")
    # print("-" * 70)

    # optimizer = PortfolioOptimizer()
    # correlations = optimizer.calculate_correlations(all_forecasts)
    # optimizer.print_summary(correlations)

    # # Combine forecasts
    # combined_forecast = optimizer.combine_forecasts(all_forecasts)

    # # Print final summary table
    # print("\n" + "=" * 70)
    # print("STRATEGY PERFORMANCE SUMMARY")
    # print("=" * 70)

    # summary_data = {}
    # for label, result in all_results.items():
    #     metrics = result['metrics']
    #     if metrics:
    #         summary_data[label] = {
    #             'Sharpe': metrics['sharpe_ratio'],
    #             'Return': metrics['total_return'],
    #             'MaxDD': metrics['max_drawdown'],
    #             'WinRate': metrics['win_rate'],
    #             'Trades': metrics['num_trades'],
    #             'PnL': metrics['annual_return'] * CAPITAL,
    #         }

    # summary_df = pd.DataFrame(summary_data).T
    # print(summary_df.round(3))

    forecasts_df = pd.DataFrame(all_forecasts)

    # Calculate pairwise forecast correlations (Carver's preference for FDM)
    forecast_correlations = forecasts_df.corr()

    # Plotting correlation matrix across variations 
    plt.figure(figsize=(12,10))
    sns.heatmap(forecast_correlations, annot=True)
    plt.title('Forecast Correlations between EWMA variations')
    plt.tight_layout()
    plt.show()

    # , combined_forecast -> insert this when portfolio optimiser is confirmed 
    return all_results, correlations


if __name__ == "__main__":
    results, correlations, combined = main()
    print("\n Backtest completed")
