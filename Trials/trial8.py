import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def lagged_corr(a, b, max_lag=20):
    """
    Compute the maximum correlation between two time series considering different lags.
    
    Args:
        a, b: numpy arrays of the same length containing stock price data.
        max_lag: Maximum number of lags to consider (default is 20).
        
    Returns:
        float: Maximum correlation coefficient.
        int: Lag at which the maximum correlation occurs.
    """
    a, b = np.array(a), np.array(b)
    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag < 0:
            corr, _ = pearsonr(a[:lag], b[-lag:])  # Shift left
        elif lag > 0:
            corr, _ = pearsonr(a[lag:], b[:-lag])  # Shift right
        else:
            corr, _ = pearsonr(a, b)  # No shift
        correlations.append(corr)
    
    max_corr = max(correlations, key=abs)
    best_lag = lags[correlations.index(max_corr)]
    
    return max_corr, best_lag

# Load stock data
file_path = 'D:\Projects\Mini Project\StockPrediction\Outputs\csv\stocks.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])

# Set Date as index and handle missing values
data.set_index('Date', inplace=True)
data.interpolate(method='linear', inplace=True)  # Interpolating missing values

# Debugging: Check if there are any NaN values remaining
print("Remaining NaN values:", data.isnull().sum().sum())  # Should print 0

# Get list of stock tickers (columns)
tickers = data.columns.tolist()
results = []

# Compute lagged correlations for all stock pairs
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        stock_a, stock_b = data.iloc[:, i], data.iloc[:, j]
        corr, lag = lagged_corr(stock_a.values, stock_b.values)
        results.append((tickers[i], tickers[j], corr, lag))

# Convert results into DataFrame
df_results = pd.DataFrame(results, columns=['Stock A', 'Stock B', 'Correlation', 'Lag'])

# Categorize correlation types
df_results['Category'] = df_results['Correlation'].apply(
    lambda x: 'Strong Positive' if x > 0.5 else ('Strong Negative' if x < -0.5 else 'Weak/No Correlation')
)

# Save categorized results
df_results.to_csv('categorized_correlations.csv', index=False)

# Filter strongly correlated pairs for further analysis
strong_positive = df_results[df_results['Category'] == 'Strong Positive'].sort_values(by='Correlation', ascending=False)
strong_negative = df_results[df_results['Category'] == 'Strong Negative'].sort_values(by='Correlation')

# Print summaries
print("\nStrong Positive Correlations (corr > 0.5):")
print(strong_positive.head(10))

print("\nStrong Negative Correlations (corr < -0.5):")
print(strong_negative.head(10))

# Create a correlation matrix
correlation_matrix = pd.DataFrame(index=tickers, columns=tickers)

for stock_a, stock_b, corr, _ in results:
    correlation_matrix.loc[stock_a, stock_b] = corr
    correlation_matrix.loc[stock_b, stock_a] = corr  # Symmetric matrix

# Convert to float and save
correlation_matrix = correlation_matrix.astype(float)
correlation_matrix.to_csv('stock_correlation_matrix.csv')

# Create a mask for the heatmap
mask = np.eye(len(tickers), dtype=bool)

# Heatmap Visualization
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask, annot=False)
plt.title('Stock Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# Scatter plot for top correlated pairs
top_pairs = strong_positive.head(5)

plt.figure(figsize=(10, 6))
for _, row in top_pairs.iterrows():
    stock_a, stock_b = row['Stock A'], row['Stock B']
    plt.scatter(data[stock_a], data[stock_b], label=f"{stock_a} vs {stock_b}")

plt.xlabel("Stock A Price")
plt.ylabel("Stock B Price")
plt.legend()
plt.title("Scatter Plot of Top Correlated Stocks")
plt.savefig('top_correlation_scatter.png', dpi=300)
plt.close()

# Rolling Window Correlation as an alternative approach
rolling_corr = data.rolling(window=30).corr(pairwise=True)
rolling_corr.to_csv('rolling_correlation.csv')

print("\nAnalysis complete. Output files saved:")
print("1. stock_correlation_matrix.csv - Full correlation matrix")
print("2. categorized_correlations.csv - Categorized correlations")
print("3. correlation_heatmap.png - Heatmap visualization")
print("4. top_correlation_scatter.png - Scatter plot of top correlated stocks")
print("5. rolling_correlation.csv - Rolling correlation values")
