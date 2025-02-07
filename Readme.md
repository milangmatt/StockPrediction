# PredStock

## Description
This project analyzes the correlation between the closing prices of the top 100 Indian stocks listed on the National Stock Exchange (NSE) over a specified time period. The analysis focuses on identifying how stock prices are correlated over different lagged time intervals.

## Functionality
- **Lagged Correlation Calculation**: The code computes the Pearson correlation coefficient between pairs of stock prices, considering various lagged time intervals. This allows for the assessment of how the price of one stock may influence another over time.
- **Data Source**: Stock price data is retrieved from Yahoo Finance using the `yfinance` library, ensuring up-to-date and accurate financial information.
- **Visualization**: The results of the correlation analysis are visualized using heatmaps, which provide a clear representation of the correlation coefficients and the corresponding lags between the stocks.

## Usage
To run the analysis, ensure that the required libraries are installed and execute the script. The output will include correlation and lag heatmaps saved in the specified output directory.

## Dependencies
- numpy
- pandas
- yfinance
- scipy
- matplotlib
- seaborn