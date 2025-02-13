import yfinance as yf
import pandas as pd
import time

tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'MARUTI.NS',
    'TATAMOTORS.NS', 'HEROMOTOCO.NS', 'INFY.NS', 'WIPRO.NS',
    'HCLTECH.NS', 'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'RELIANCE.NS', 'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS'
]

def fetch_stock_data():
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            stock_data.append({
                'Ticker': ticker,
                'Last Price': round(data['Close'].iloc[-1], 2),
                'Open': round(data['Open'].iloc[-1], 2),
                'High': round(data['High'].iloc[-1], 2),
                'Low': round(data['Low'].iloc[-1], 2),
                'Volume': int(data['Volume'].iloc[-1])
            })
    return stock_data

def display_stock_data():
    while True:
        stock_data = fetch_stock_data()
        df = pd.DataFrame(stock_data)
        print(df.to_string(index=False))
        print("\nUpdating in 1 seconds...\n")
        time.sleep(1)

if __name__ == "__main__":
    display_stock_data()
