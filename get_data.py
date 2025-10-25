import ccxt
import pandas as pd
from datetime import datetime
import os

output_dir = r"C:\Users\diste\OneDrive\Desktop\Data\Crypto"

# get_data retrieves data and saves it to our output directory stated above 

def get_data(symbol, timeframe, limit=1000, output_dir = output_dir):
    
    """
    Fetch data from Binance using CCXT

    Parameters:
    - symbol: Trading pair (e.g 'BTC/USDT', 'ETH/USDT')
    - timeframe: Candlestick timeframe (default: '1h')
    - limit: Number of candles to fetch (default: 1000)
    - output_dir: Directory to save the CSV file

    Returns:
    - DataFrame with OHLCV data
    """

    # Initialize Binance exchange
    exchange = ccxt.binance()

    print(f"Fetching {symbol} data from Binance...")
    print(f"Timeframe: {timeframe}, Limit: {limit} candles")

    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"Successfully fetched {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        filename = f"{output_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(filename)
        print(f"Data saved to: {filename}")

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 'BNB/USDT', 'HYPE/USDT']
timeframes = ['1d']

limit = 1000 
# Loop through assets 
for asset in assets:

    # Loop through different timeframes 
    for timeframe in timeframes: 
        
        get_data(asset, timeframe, limit, output_dir) # get data and save to data directory 