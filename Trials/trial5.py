import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

    max_lag = 20  # Maximum lag to consider
    
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

# Read the stock data from CSV file with the specific format provided
data = pd.read_csv('D:\Mikku\Miniproject\StockPrediction\Outputs\csv\data_2025-02-27_14-47-01.csv', parse_dates=['Date'])

# Set the Date column as index
data.set_index('Date', inplace=True)

# Get the list of stock tickers (column names)
tickers = data.columns.tolist()

results = []

# Calculate lagged correlations for all pairs of stocks
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        stock_a = data.iloc[:, i]
        stock_b = data.iloc[:, j]
        corr, lag = laggedCorr(stock_a.values, stock_b.values)
        results.append((tickers[i], tickers[j], corr, lag))

# Display the results
print("All Lagged Correlations:")
for stock_a, stock_b, corr, lag in results:
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

# Identify negatively correlated pairs
negatively_correlated = [(stock_a, stock_b, corr, lag) for stock_a, stock_b, corr, lag in results if corr < -0.5]
positively_correlated = [(stock_a, stock_b, corr, lag) for stock_a, stock_b, corr, lag in results if corr > 0.5]

print("\nStrongly Negatively Correlated Pairs (correlation < -0.5):")
for stock_a, stock_b, corr, lag in sorted(negatively_correlated, key=lambda x: x[2]):
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

print("\nStrongly Positively Correlated Pairs (correlation > 0.5):")
for stock_a, stock_b, corr, lag in sorted(positively_correlated, key=lambda x: x[2], reverse=True):
    print(f"{stock_a} and {stock_b}: Correlation = {corr:.2f}, Lag = {lag} days")

# Create a DataFrame for correlation results
correlation_matrix = pd.DataFrame(index=tickers, columns=tickers)

# Fill the correlation matrix
for stock_a, stock_b, corr, lag in results:
    correlation_matrix.loc[stock_a, stock_b] = corr
    correlation_matrix.loc[stock_b, stock_a] = corr  # Symmetric matrix

# Set diagonal values to 1.0 (correlation of stock with itself)
for ticker in tickers:
    correlation_matrix.loc[ticker, ticker] = 1.0

# Convert the correlation matrix to numeric
correlation_matrix = correlation_matrix.astype(float)

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv('stock_correlation_matrix.csv')

# Plot the correlation heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            xticklabels=True, yticklabels=True)
plt.title('Correlation Heatmap between NSE Stocks')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# Find the top 10 most correlated and anti-correlated pairs
flat_corr = []
for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        flat_corr.append((tickers[i], tickers[j], correlation_matrix.iloc[i, j]))

# Sort by absolute correlation
top_correlations = sorted(flat_corr, key=lambda x: abs(x[2]), reverse=True)[:20]

# Plot lag vs correlation scatter plot for selected pairs
plt.figure(figsize=(12, 8))
for stock_a, stock_b, corr, lag in results[:100]:  # Plot first 100 pairs to avoid overcrowding
    plt.scatter(lag, corr, alpha=0.5)

plt.title('Lagged Correlations between Top NSE Stocks')
plt.xlabel('Lag (days)')
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='grey', lw=1, ls='--')
plt.axvline(0, color='grey', lw=1, ls='--')
plt.grid(True)
plt.savefig('lag_correlation_scatter.png', dpi=300)
plt.close()

# Create a visualization of the top correlated pairs
plt.figure(figsize=(14, 10))

top_corr_df = pd.DataFrame(top_correlations, columns=['Stock A', 'Stock B', 'Correlation'])
top_corr_df['AbsCorrelation'] = top_corr_df['Correlation'].abs()
top_corr_df.sort_values('AbsCorrelation', ascending=False, inplace=True)

colors = ['green' if c > 0 else 'red' for c in top_corr_df['Correlation']]
plt.barh(
    [f"{a} - {b}" for a, b, _ in zip(top_corr_df['Stock A'], top_corr_df['Stock B'])],
    top_corr_df['Correlation'],
    color=colors
)
plt.title('Top 20 Stock Correlations by Absolute Value')
plt.xlabel('Correlation Coefficient')
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300)
plt.close()

print("\nAnalysis complete. Output files saved:")
print("1. stock_correlation_matrix.csv - Full correlation matrix")
print("2. correlation_heatmap.png - Heatmap visualization of correlations")
print("3. lag_correlation_scatter.png - Scatter plot of lag vs correlation")
print("4. top_correlations.png - Bar chart of top 20 correlations")