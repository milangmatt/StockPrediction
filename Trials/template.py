# Topic 1 – Stock data and lagged correlation
# Goal:  Out from the data of the top 100 Indian stocks, are there any stocks that appear to have a strong lagged correlation?
# Task 1 – Understand my meaning of “lagged correlation”.
# Task 2 – Create an algorithm that can measure the lagged correlation of two data series.
# Task 3 – Verify your algorithm in Task 2 is correct, using synthetic data.
# Task 4 – Download the closing value of the top 100 Indian stocks.
# Task 5 – Apply your algorithm to all pairs of the 100 stocks
# Task 6 – Which pairs of stocks are strongly lagged correlated?

import numpy as np


def laggedCorr(a, b, max_lag=10):
    """
    Computes the lagged correlation between two time series.

    Parameters:
        a (numpy array): First time series.
        b (numpy array): Second time series.
        max_lag (int): Maximum lag to consider.

    Returns:
        max_corr (float): Maximum correlation found.
        best_lag (int): The lag value at which max correlation is achieved.
    """
    best_lag = 0
    max_corr = -1  # Initialize with a low value

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_a = a[:lag]  # Shift left
            shifted_b = b[-lag:]
        elif lag > 0:
            shifted_a = a[lag:]  # Shift right
            shifted_b = b[:-lag]
        else:
            shifted_a, shifted_b = a, b  # No shift

        # Compute correlation only if the lengths match
        if len(shifted_a) == len(shifted_b):
            corr = np.corrcoef(shifted_a, shifted_b)[0, 1]
            if abs(corr) > abs(max_corr):  # Track the max correlation
                max_corr = corr
                best_lag = lag

    return max_corr, best_lag



N = 100

def testCase(k):
    if k == 0:
        # No relationship, expect correlation close to 0
        a = np.random.rand(N)
        b = np.random.rand(N)
        expectedOutput = 0  # No correlation expected
        return (a, b, expectedOutput)

    elif k == 1:
        # 'b' is a lagged version of 'a' by 3 steps
        a = np.random.rand(N)
        b = np.concatenate((a[3:], a[:3]))  # Shift 'a' by 3
        expectedOutput = 0.98  # Correlation close to 1 expected
        return (a, b, expectedOutput)


#=========================
# MAIN
#=========================

for testId in range(2):
    a, b, expectedOutput = testCase(testId)
    out, best_lag = laggedCorr(a, b)
    print(f"Test case {testId}: Expected ~{expectedOutput}, Got {out:.3f} at lag {best_lag}")