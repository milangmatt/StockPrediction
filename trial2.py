import numpy as np
from scipy import stats

def laggedCorr(a, b):
    """
    Calculate the maximum correlation between two series considering different lags.
    
    Args:
        a, b: numpy arrays of same length containing the time series data
        
    Returns:
        float: Maximum correlation coefficient found across all lags
    """
    # Convert inputs to numpy arrays if they aren't already
    a = np.array(a)
    b = np.array(b)
    
    # Get length of series
    n = len(a)
    max_lag = min(n // 4, 20)  # Look at lags up to 1/4 of series length or 20, whichever is smaller
    
    # Store correlations for different lags
    correlations = []
    
    # Calculate correlation for different lags
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # b is shifted forward
            corr = stats.pearsonr(a[-lag:], b[:lag])[0]
        elif lag > 0:
            # a is shifted forward
            corr = stats.pearsonr(a[:-lag], b[lag:])[0]
        else:
            # no shift
            corr = stats.pearsonr(a, b)[0]
        correlations.append(corr)
    
    # Return the maximum absolute correlation found
    return max(correlations, key=abs)

N = 100
def testCase(k):
    if k == 0:
        # no relationship, expect zero
        a = np.random.rand(N)
        b = np.random.rand(N)
        expectedOutput = 0
        return (a, b, expectedOutput)
    elif k == 1:
        # a is lagged behind b always, expect an output close to 1
        a = np.random.rand(N)
        b = np.concatenate((a[3:N], a[0:3]))
        expectedOutput = 0.98  # just a number close to 1
        return (a, b, expectedOutput)
    elif k == 2:
        # Perfect negative correlation with lag
        a = np.random.rand(N)
        b = -np.concatenate((a[5:N], a[0:5]))
        expectedOutput = -0.98
        return (a, b, expectedOutput)
    elif k == 3:
        # Moderate positive correlation with noise
        a = np.random.rand(N)
        noise = np.random.normal(0, 0.2, N)
        b = np.concatenate((a[2:N], a[0:2])) + noise
        expectedOutput = 0.7
        return (a, b, expectedOutput)
        
# Main execution
for testId in range(4):
    (a, b, expectedOutput) = testCase(testId)
    out = laggedCorr(a, b)
    print(f"For test case {testId}, the expected output is {expectedOutput:.2f}, the output is {out:.2f}")