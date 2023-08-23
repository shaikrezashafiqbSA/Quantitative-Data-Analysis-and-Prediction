
import numpy as np
import numba as nb
from tqdm import tqdm


import pandas_ta as pta


@nb.njit(cache=True)
def rolling_tanh(x, window):
    np_tanh_atr = np.full(len(x),np.nan)
    for i in range(len(x)):
        if i < window:
            continue
        else:
            np_tanh_atr[i] = np.tanh(x[i-window:i])[-1]#*1000
    return np_tanh_atr


def normalize(x, newRange=(1,100), window=288):
    np_lev = np.full(len(x),np.nan)
    for i in range(len(x)):
        if i < window:
            continue
        else:
            x_i = x[i-window:i]
            xmin, xmax = np.min(x_i),np.max(x_i)
            norm = (x_i-xmin)/(xmax-xmin)
            
        if newRange == (0, 1):
            new_norm = norm # wanted range is the same as norm
        elif newRange != (0, 1):
            new_norm = norm * (newRange[1] - newRange[0]) + newRange[0] #scale to a different range.
        np_lev[i] = new_norm[-1]
    return np_lev
        

def calc_position_sizes(df, high="high", low="low",close="close", window_ATR=288, window_norm=288, newRange=(1,100)):
    """
    This function populates df with tanh atr of <<window>> lookback
    """
    
    df["ATR"] = pta.atr(high=df[high],low=df[low],close=df[close], length=window_ATR)
    # df["ATR"]=df["ATR"]/df[close]
    df["size"] = df["ATR"]/df[close] * 100
    df["size"] = normalize(df["size"].values, newRange = newRange, window=window_norm)
    return df