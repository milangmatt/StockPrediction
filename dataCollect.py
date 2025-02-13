import pandas as pd
import yfinance as yf
from datetime import datetime

# List of top 100 Indian stocks (NSE tickers)
tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'MARUTI.NS',
    'TATAMOTORS.NS', 'HEROMOTOCO.NS', 'INFY.NS', 'WIPRO.NS',
    'HCLTECH.NS', 'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'RELIANCE.NS', 'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS',
    # Add more tickers to reach 100
]

# Download the closing prices
data = yf.download(tickers, start='2020-01-04', end='2025-02-12')['Close']

# Generate a dynamic file name with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"Outputs/csv/data_{timestamp}.csv"

# Save to CSV with the dynamic filename
data.to_csv(file_name)

print(f"Data saved as {file_name}")
