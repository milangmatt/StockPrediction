import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

def laggedCorr(a, b):
    """
    Calculate the maximum correlation between two series considering different lags.
    
    Args:
        a, b: numpy arrays of same length containing the time series data
        
    Returns:
        float: Maximum correlation coefficient found across all lags
        int: Lag at which the maximum correlation occurs
    """
    a = np.array(a)
    b = np.array(b)
  
    n = len(a)

    max_lag = min(n // 4, 20)  # Maximum lag to consider
    
    correlations = []
    lags = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = stats.pearsonr(a[-lag:], b[:lag])[0]
        elif lag > 0:
            corr = stats.pearsonr(a[:-lag], b[lag:])[0]
        else:
            corr = stats.pearsonr(a, b)[0]
        correlations.append(corr)
        lags.append(lag)
    
    max_corr = max(correlations, key=abs)
    max_lag_index = correlations.index(max_corr)
    
    return max_corr, lags[max_lag_index]

# List of top 100 Indian stocks (NSE tickers)
tickers = [
    'HDFCBANK.NS',
    'ICICIBANK.NS',
    'KOTAKBANK.NS',
    'AXISBANK.NS',
    'MARUTI.NS',
    'TATAMOTORS.NS',
    'HEROMOTOCO.NS',
    'TCS.NS',
    'INFY.NS',
    'WIPRO.NS',
    'HCLTECH.NS',
    'HINDUNILVR.NS',
    'ITC.NS',
    'NESTLEIND.NS',
    'BRITANNIA.NS',
    'RELIANCE.NS',
    'NTPC.NS',
    'TATAPOWER.NS',
    'TITAN.NS',
    # Add more tickers to reach 100
]

# Download the closing prices
data = yf.download(tickers, start='2025-01-04', end='2025-02-04')['Close']

results = []

# Calculate lagged correlations for all pairs of stocks
for i in range(len(data.columns)):
    for j in range(i + 1, len(data.columns)):
        stock_a = data.iloc[:, i]
        stock_b = data.iloc[:, j]
        corr, lag = laggedCorr(stock_a.values, stock_b.values)
        results.append((data.columns[i], data.columns[j], corr, lag))

# Display the results
print("All Lagged Correlations:")
for stock_a, stock_b, corr, lag in results:
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

# Identify negatively correlated pairs
negatively_correlated = [(stock_a, stock_b, corr, lag) for stock_a, stock_b, corr, lag in results if corr < 0]

positively_correlated = [(stock_a, stock_b, corr, lag) for stock_a, stock_b, corr, lag in results if corr >= 0]

print("\nNegatively Correlated Pairs:")
for stock_a, stock_b, corr, lag in negatively_correlated:
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

print("\nPositively Correlated Pairs:")
for stock_a, stock_b, corr, lag in positively_correlated:
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

# Plotting the correlations
plt.figure(figsize=(12, 6))
for stock_a, stock_b, corr, lag in results:
    plt.scatter(lag, corr, alpha=0.5, label=f'{stock_a} vs {stock_b}' if lag == 0 else "")

plt.title('Lagged Correlations between Top NSE Stocks')
plt.xlabel('Lag (days)')  # Labeling the x-axis as days
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='grey', lw=1, ls='--')
plt.axvline(0, color='grey', lw=1, ls='--')
plt.grid()
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()