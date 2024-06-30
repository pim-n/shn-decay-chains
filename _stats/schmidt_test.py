from bisect import bisect_left

import numpy as np
import scipy as scp
import pandas as pd

def schmidt_test(decay_times):
    """
    Calculates σ_Θ_exp following the original Schmidt test.

    Keyword arguments:
        decay_times         1D array of decay times to be tested.

    Returns:
        sigma_theta_exp     The σ_Θ_exp value for the given data.
        confidence_interval The upper and lower limit to the confidence interval.

    """
    
    sigma_theta_exp = np.nanstd(np.log(decay_times))

    if len(decay_times) <= 100: # Schmidt (2000) has tabulated values for n <= 100 events
        dat = pd.read_pickle("_stats/st_sigma_theta_exp_properties.pkl") # load tabulated values
        i = bisect_left(dat.loc[:,'n'], len(decay_times)) # find the closest value in tabulated values
        lim_l = dat.loc[i,'lower limit of σ_Θ_exp']
        lim_h = dat.loc[i,'upper limit of σ_Θ_exp']
    else: # if n > 100, calculate limits based on analytical formula
        lim_l = 1.28-2.15/np.sqrt(len(decay_times))
        lim_h = 1.28+2.15/np.sqrt(len(decay_times))

    confidence_interval = (lim_l, lim_h)
    return sigma_theta_exp, confidence_interval

def g_nan_mean(data):
    """
    Code written by Anton
    """

    if len(np.shape(data)) == 1:
        return data
    ret = np.empty(np.shape(data)[0])
    for i in range(np.shape(data)[0]):
        temp = 1
        steps = 0
        for j in range(np.shape(data)[1]):
            if isinstance(data, pd.DataFrame) and ~np.isnan(data.iloc[i,j]):
                temp *= data.iloc[i,j]
            elif not isinstance(data, pd.DataFrame) and ~np.isnan(data[i,j]):
                temp *= data[i,j]
            else:
                break
            steps += 1
        ret[i] = temp**(1./steps)
    return ret

def generalised_schmidt_test(df):
    """
    Generalised Schmidt Test

    Keyword arguments:
        df    DataFrame containing simulated chain event times  
    """

    if any(df.iloc[:,-1].str.contains('SF')):
        df = df.iloc[:,:-1]

    theta = np.log(df)
    theta_var = np.square(theta - np.nanmean(theta, axis=0))
    gen_Schmidt_temp = g_nan_mean(theta_var)
    sigma_theta_exp = np.sqrt(np.mean(gen_Schmidt_temp))

    if len(df) <= 100: # Schmidt (2000) has tabulated values for n <= 100 events
        dat = pd.read_pickle("_stats/st_sigma_theta_exp_properties.pkl") # load tabulated values
        i = bisect_left(dat.loc[:,'n'], len(df)) # find the closest value in tabulated values
        lim_l = dat.loc[i,'lower limit of σ_Θ_exp']
        lim_h = dat.loc[i,'upper limit of σ_Θ_exp']
    else: # if n > 100, calculate limits based on analytical formula
        lim_l = 1.28-2.15/np.sqrt(len(df))
        lim_h = 1.28+2.15/np.sqrt(len(df))

    confidence_interval = (lim_l, lim_h)

    return sigma_theta_exp, confidence_interval