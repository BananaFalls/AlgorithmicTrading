# Core computation 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Performance metrics and reporting 
# import quantstats as qs 

# def ewma_strategy(prices, fast, slow, vol_target, capital):

#     # Step 1: Calculate  EWMA
#     fast_ewma = prices.ewm(span = fast, adjust = False).mean()
#     slow_ewma = prices.ewm(span = slow, adjust = False).mean()

#     # Step 2: Get raw signal
#     raw_signal = (fast_ewma - slow_ewma) / prices

#     # Step 3: Scale to forecast between -20 to 20
#     scaling_factor = 10 / raw_signal.abs().mean()
#     forecast = (raw_signal * scaling_factor).clip(-20,20)

#     # Step 4: Calculate volatility
#     returns = prices.pct_change()
#     volatility = returns.ewm(span = 25).std() * np.sqrt(256)

#     # Step 5: Position sizing
#     instrument_currency_vol = volatility * prices # asset's vol in dollar terms
#     notional_position = (vol_target * capital ) / instrument_currency_vol # desired vol of portfolio in dollar terms
#     positions = notional_position * (forecast/10)

#     # Step 6: Calculating returns
#     strategy_returns = positions.shift(1) * returns

#     return forecast.dropna(), strategy_returns.dropna(), positions.dropna()


def rolling_window_backtest(prices, fast, slow, vol_target, capital, train_window=252, test_window=63):
    """
    Backtest strategy using rolling window (walk-forward analysis)

    Parameters:
    - prices: Price series
    - fast, slow: EWMA parameters
    - vol_target, capital: Strategy parameters
    - train_window: Days to train on (default 252 = 1 year)
    - test_window: Days to test on (default 63 = 3 months)

    Returns:
    - Dictionary with forecasts, returns, and positions for the entire period
    """

    all_forecasts = []
    all_returns = []
    all_positions = []
    timestamps = []

    # Walk forward through the data
    for i in range(train_window, len(prices) - test_window, test_window):
        # Training period
        train_prices = prices.iloc[i-train_window:i]

        # Test period
        test_prices = prices.iloc[i:i+test_window]
        test_returns = test_prices.pct_change()

        # Apply strategy on training data
        fast_ewma_train = train_prices.ewm(span=fast, adjust=False).mean()
        slow_ewma_train = train_prices.ewm(span=slow, adjust=False).mean()
        raw_signal_train = (fast_ewma_train - slow_ewma_train) / train_prices

        # Calculate scaling factor from training data 
        scaling_factor = 10 / raw_signal_train.abs().mean()

        # Apply to test data
        fast_ewma_test = test_prices.ewm(span=fast, adjust=False).mean()
        slow_ewma_test = test_prices.ewm(span=slow, adjust=False).mean()
        raw_signal_test = (fast_ewma_test - slow_ewma_test) / test_prices
        forecast = (raw_signal_test * scaling_factor).clip(-20, 20)

        # Calculate volatility target and positions
        volatility = test_returns.ewm(span=25).std() * np.sqrt(256)
        instrument_currency_vol = volatility * test_prices
        notional_position = (vol_target * capital) / instrument_currency_vol
        positions = notional_position * (forecast / 10)

        # Calculate returns
        strategy_returns = positions.shift(1) * test_returns

        # Store results
        all_forecasts.extend(forecast.dropna().values)
        all_returns.extend(strategy_returns.dropna().values)
        all_positions.extend(positions.dropna().values)
        timestamps.extend(strategy_returns.dropna().index)

    return {
        'forecasts': pd.Series(all_forecasts, index=timestamps),
        'returns': pd.Series(all_returns, index=timestamps),
        'positions': pd.Series(all_positions, index=timestamps)
    }

# Different variations of strategy
ewma_variations = [
    # Very fast
    (4, 16), (8, 32),
    # Fast
    (16, 64), (24, 96),
    # Medium
    (32, 128), (48, 192),
]

# Calculate forecasts for all variations
all_forecasts = {}

# Read data, convert into csv 
df_prices = pd.read_csv(r'C:\Users\diste\OneDrive\Desktop\Data\Crypto\BTC_USDT_1d.csv')
df_close = df_prices['close']

# Select volatility target and capital 
vol_target = 0.25
capital = 10000

# Loop through different variations 
for fast, slow in ewma_variations:
    label = f"EWMA_{fast}_{slow}"
    returns, forecast = rolling_window_backtest(df_close, fast, slow, vol_target, capital)
    all_forecasts[label] = forecast

