import numpy as np
import numba as nb



@nb.njit(cache=True)
def calc_sr(x):
    x = x[~np.isnan(x)]
    if len(x) >0:
        x_std = x.std()
        x_mean = x.mean()
        if np.isnan(x_std) or np.isnan(x_mean):
            sr = 0
        elif x_std == 0.0:
            sr = 0
        else:
            sr = x_mean/x_std
        return sr
    else:
        return 0

@nb.njit(cache=True)
def calc_rolling_sr(ret, window, buffer=120):
    # print(ret)
    n=len(ret)
    np_rolling_sr = np.full(n,np.nan)
    
    for i in range(n):
        if (i <window) or (np.mod(i,buffer) != 0):
            continue
        x = ret[i-window:i+1]
        np_rolling_sr[i] = calc_sr(x)
        
    return np_rolling_sr