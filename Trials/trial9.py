import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Define sectors for each company
df_sectors = {
    "Financials": ["SHRIRAMFIN.NS", "BAJAJHLDNG.NS", "CHOLAFIN.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "INDUSINDBK.NS", "HDFCBANK.NS", "AXISBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "BANKBARODA.NS", "PNB.NS", "UNIONBANK.NS"],
    "Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS", "TORNTPHARM.NS", "CIPLA.NS", "DRREDDY.NS", "ZYDUSLIFE.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Consumer Goods": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "TATACONSUM.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS", "UNITDSPR.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "GAIL.NS", "NTPC.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "JSWENERGY.NS"],
    "Automobiles": ["TATAMOTORS.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "MOTHERSON.NS"],
    "Infrastructure": ["L&T.NS", "DLF.NS", "GRASIM.NS", "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "BHEL.NS"],
    "Metals": ["HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "JINDALSTEL.NS", "VEDL.NS"],
    "Telecommunications": ["BHARTIARTL.NS", "JIOFIN.NS"],
    "Miscellaneous": ["ZOMATO.NS", "NAUKRI.NS", "INDIGO.NS", "TRENT.NS", "IRCTC.NS", "LODHA.NS", "VBL.NS"]
}

# Load stock price data
data = pd.read_csv('D:\Projects\Mini Project\StockPrediction\Outputs\csv\stocks.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

def laggedCorr(a, b, max_lag=20):
    """Calculate the maximum correlation between two series considering different lags."""
    a, b = np.array(a), np.array(b)
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

def calculate_sector_correlation(data, sector_stocks):
    """Calculates lagged correlation matrix for a given sector."""
    sector_data = data[sector_stocks].dropna()
    results = []
    for i in range(len(sector_stocks)):
        for j in range(i + 1, len(sector_stocks)):
            stock_a, stock_b = sector_stocks[i], sector_stocks[j]
            corr, lag = laggedCorr(sector_data[stock_a], sector_data[stock_b])
            results.append((stock_a, stock_b, corr, lag))
    return results

# Calculate and save correlation matrices for each sector
for sector, stocks in df_sectors.items():
    available_stocks = [stock for stock in stocks if stock in data.columns]
    if available_stocks:
        lagged_results = calculate_sector_correlation(data, available_stocks)
        
        # Save results
        df_lagged = pd.DataFrame(lagged_results, columns=['Stock A', 'Stock B', 'Correlation', 'Lag'])
        df_lagged.to_csv(f'D:\Projects\Mini Project\StockPrediction\Outputs\{sector}_lagged_correlation.csv', index=False)
        
        # Plot heatmap
        correlation_matrix = df_lagged.pivot(index='Stock A', columns='Stock B', values='Correlation')
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
        plt.title(f'Lagged Correlation Matrix - {sector}')
        plt.savefig(f'D:\Projects\Mini Project\StockPrediction\Outputs\{sector}_lagged_correlation_heatmap.png')
        plt.close()

print("Sector-wise lagged correlation analysis complete. Files saved.")
