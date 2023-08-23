import numpy as np
import pandas as pd

def calc_sigmoid(z):
    """
    The sigmoid function.
    Args:
        z: float, vector, matrix
    Returns:
        sigmoid: float
    """
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

def calc_tanh(z):
    return np.tanh(z)          
        

def rolling_quantiles(df0, 
                      target_signal,
                      L_q_lookback,
                      L_q_upper,
                      L_q_lower,
                      S_q_lookback,
                      S_q_upper,
                      S_q_lower):
    # drop where sig is nan
    df1 = df0.copy()
    df = df1.dropna(subset=[target_signal]).copy()
    
    df['L_uq'] = df[target_signal].rolling(L_q_lookback).quantile(L_q_upper).shift(1)
    df['L_lq'] = df[target_signal].rolling(L_q_lookback).quantile(L_q_lower).shift(1)
    
    df['S_uq'] = df[target_signal].rolling(S_q_lookback).quantile(S_q_upper).shift(1)
    df['S_lq'] = df[target_signal].rolling(S_q_lookback).quantile(S_q_lower).shift(1)
    df1= pd.merge(df1,df[["L_uq","L_lq","S_uq","S_lq"]], right_index=True, left_index=True, how="left")
    
    return df1


def rolling_std(df0, 
                      target_signal,
                      L_q_lookback,
                      L_q_upper,
                      L_q_lower,
                      S_q_lookback,
                      S_q_upper,
                      S_q_lower):
    # drop where sig is nan
    df1 = df0.copy()
    df = df1.dropna(subset=[target_signal]).copy()
    
    stdev = 2*df[target_signal].rolling(L_q_lookback).std().shift(1)
    
    df['L_uq'] = df[target_signal].rolling(L_q_lookback).quantile(0.5).shift(1) + stdev
    df['L_lq'] = df[target_signal].rolling(L_q_lookback).quantile(0.5).shift(1) - stdev
    
    df['S_uq'] = df[target_signal].rolling(S_q_lookback).quantile(0.5).shift(1) + stdev
    df['S_lq'] = df[target_signal].rolling(S_q_lookback).quantile(0.5).shift(1) - stdev
    df1= pd.merge(df1,df[["L_uq","L_lq","S_uq","S_lq"]], right_index=True, left_index=True, how="left")
    
    return df1

    
def calc_signal_thresholds(sig_dict,
                           normalize=False,
                           target_signal = "comp_sig",
                           threshold_func = "qtl",
                           L_q_lookback = 288,
                           L_q_upper = 0.95, 
                           L_q_lower = 0.05,
                           S_q_lookback = 288,
                           S_q_upper = 0.95,
                           S_q_lower = 0.05):
    for timeframe, df in sig_dict.items():
        
        
        if normalize == "sigmoid":
            df[target_signal] = calc_sigmoid(df[target_signal].values)
        elif normalize == "tanh":
            df[target_signal] = calc_tanh(df[target_signal].values)
            
            
        # df[f'L_uq'] = df[target_signal].rolling(L_q_lookback).quantile(L_q_upper).shift(1)
        # df[f'L_lq'] = df[target_signal].rolling(L_q_lookback).quantile(L_q_lower).shift(1)
        
        # df[f'S_uq'] = df[target_signal].rolling(S_q_lookback).quantile(S_q_upper).shift(1)
        # df[f'S_lq'] = df[target_signal].rolling(S_q_lookback).quantile(S_q_lower).shift(1)
        if threshold_func == "qtl":
            
            df = rolling_quantiles(df,
                                   target_signal=target_signal,
                                   L_q_lookback=L_q_lookback,
                                   L_q_upper=L_q_upper,
                                   L_q_lower=L_q_lower,
                                   S_q_lookback=S_q_lookback,
                                   S_q_upper=S_q_upper,
                                   S_q_lower=S_q_lower)
        elif threshold_func == "std":
            df = rolling_std(df,
                            target_signal=target_signal,
                            L_q_lookback=L_q_lookback,
                            L_q_upper=L_q_upper,
                            L_q_lower=L_q_lower,
                            S_q_lookback=S_q_lookback,
                            S_q_upper=S_q_upper,
                            S_q_lower=S_q_lower)
        
        sig_dict[timeframe] = df
        
    return sig_dict