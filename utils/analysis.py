import numpy as np
from scipy import stats



def compute_population_stats(responses, alpha=0.05):

    # Sample size, mean, and standard error of the mean
    n = len(responses)
    mean_val = np.mean(responses)
    std_val = np.std(responses, ddof=1)
    sem = std_val / np.sqrt(n)
    
    # Two-tailed t-value for CI = 100 * (1 - alpha) %
    # degrees of freedom = n-1
    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Confidence interval
    error_margin = t_val * sem
    return mean_val, error_margin