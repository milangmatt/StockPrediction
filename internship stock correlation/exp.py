import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Function to calculate correlation vs. lag
def lagged_correlation(a, b, max_lag):
    a = np.array(a)
    b = np.array(b)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            corr = pearsonr(a[-lag:], b[:lag])[0]
        elif lag > 0:
            corr = pearsonr(a[:-lag], b[lag:])[0]
        else:
            corr = pearsonr(a, b)[0]
        correlations.append(corr)
    
    return lags, correlations

# Load stock data
data = pd.read_csv('D:/internship stock correlation/stocks.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

tickers = data.columns.tolist()
results = []

# Compute lagged correlations
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        stock_a = data.iloc[:, i]
        stock_b = data.iloc[:, j]
        corr, _ = lagged_correlation(stock_a.values, stock_b.values, max_lag=20)
        results.append((tickers[i], tickers[j], max(corr, key=abs)))

# Sort and select top 20 correlated pairs
top_correlations = sorted(results, key=lambda x: abs(x[2]), reverse=True)[:20]

# Generate and save correlation vs. lag plots for top 20 correlated pairs
max_lag_values = [20, 60, 120]

for stock_a, stock_b, _ in top_correlations:
    plt.figure(figsize=(12, 6))
    for max_lag in max_lag_values:
        lags, correlations = lagged_correlation(data[stock_a].values, data[stock_b].values, max_lag)
        plt.plot(lags, correlations, label=f"max_lag={max_lag}")
    
    plt.axhline(0, color='black', linestyle='dashed', linewidth=0.8)
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation")
    plt.title(f"Correlation vs. Lag for {stock_a} and {stock_b}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{stock_a}_{stock_b}_correlation_plot.png", dpi=300)
    plt.close()

print("Correlation vs. lag plots saved for top 20 correlated stock pairs.")
