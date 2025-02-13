import numpy as np
import pandas as pd
from nespy import NSE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def laggedCorr(a, b):
    a = np.array(a)  # stock values of company a 
    b = np.array(b)  # stock values of company b
  
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
    stock_data = nse.get_history(symbol=ticker, start='2025-01-04', end='2025-02-04')
    data[ticker] = stock_data['Close']

results = []

# Calculate lagged correlations for all pairs of stocks
for i in range(len(data.columns)):
    for j in range(i + 1, len(data.columns)):
        stock_a = data.iloc[:, i]
        stock_b = data.iloc[:, j]
        corr, lag = laggedCorr(stock_a.values, stock_b.values)
        results.append((data.columns[i], data.columns[j], corr, lag))

# Create a DataFrame for correlation results
correlation_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
lag_matrix = pd.DataFrame(index=data.columns, columns=data.columns)

# Fill the correlation and lag matrices
for stock_a, stock_b, corr, lag in results:
    correlation_matrix.loc[stock_a, stock_b] = corr
    correlation_matrix.loc[stock_b, stock_a] = corr  # Symmetric matrix
    lag_matrix.loc[stock_a, stock_b] = lag
    lag_matrix.loc[stock_b, stock_a] = lag  # Symmetric matrix

# Convert the matrices to numeric
correlation_matrix = correlation_matrix.astype(float)
lag_matrix = lag_matrix.astype(float)

# Plotting the correlation and lag heatmaps
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=axes[0], annot_kws={"size": 8,"rotation": 90})
axes[0].set_title('Correlation Heatmap between Top NSE Stocks')
axes[0].set_xlabel('Stocks')
axes[0].set_ylabel('Stocks')

# Lag heatmap
sns.heatmap(lag_matrix, annot=True, fmt=".0f", cmap='viridis', ax=axes[1], annot_kws={"size": 8,"rotation": 90})
axes[1].set_title('Lag Heatmap between Top NSE Stocks')
axes[1].set_xlabel('Stocks')
axes