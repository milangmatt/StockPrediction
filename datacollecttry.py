import pandas as pd
from nespy import NSE

# List of top 100 Indian stocks (NSE tickers)
tickers = [
    'HDFCBANK',
    'ICICIBANK',
    'KOTAKBANK',
    'AXISBANK',
    'MARUTI',
    'TATAMOTORS',
    'HEROMOTOCO',
    'TCS',
    'INFY',
    'WIPRO',
    'HCLTECH',
    'HINDUNILVR',
    'ITC',
    'NESTLEIND',
    'BRITANNIA',
    'RELIANCE',
    'NTPC',
    'TATAPOWER',
    'TITAN',
    # Add more tickers to reach 100
]

# Initialize NSE object
nse = NSE()

# Download the closing prices
data = pd.DataFrame()

for ticker in tickers:
    stock_data = nse.get_history(symbol=ticker, start='2020-01-04', end='2025-02-04')
    data[ticker] = stock_data['Close']

data.to_csv(r"data.csv")