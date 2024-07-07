
import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm
import time

from utils.find_index import find_list_index


# ============================================================================================================================================
# ============================================================================================================================================
#                               Base Zscore Function
# ============================================================================================================================================
# ============================================================================================================================================



@nb.njit(cache=True)
def rolling_zscore(np_price, 
                   np_windows, 
                   np_thresholds
                   ):
    
    n = len(np_price)
    np_z = np.full(n, np.nan)
    np_zsig = np.full(n, np.nan)

    for i in range(0,n):
        window = int(np_windows[i])
        threshold = float(np_thresholds[i])
        if window > i:
            continue
        
        start_index = int(i+1-window)
        end_index = int(i+1)


        # START SIGNAL CALCULATION
        np_price_i=np_price[start_index:end_index]
        np_price_i = np_price_i.flatten()[~np.isnan(np_price_i.flatten())] #np_price_i[~np.isnan(np_price_i)]
        ret = np.diff(np.log(np_price_i))

        if (len(ret) == 0) or (np.std(ret) == 0.0):
            res = 0
            np_z[i]=res
        else:
            res = (ret[-1] - np.mean(ret))/np.std(ret)
            np_z[i]=res

        res1 = np.sign(res) * np.floor(abs(res)) * (abs(res) >= threshold)
        np_zsig[i] = res1

    return np_z, np_zsig


# ============================================================================================================================================
# ============================================================================================================================================
#                               Zscore Functions
# ============================================================================================================================================
# ============================================================================================================================================



"""
Static Param Function
"""
@nb.njit(cache=True)
def static_params(np_klines: np.array, column_indexes: np.array):
    windows = 288
    threshold = 2
    np_windows = np.full(np_klines.shape[0], np.nan)
    np_thresholds = np.full(np_klines.shape[0], np.nan)
    for i in range(np_klines.shape[0]):
        np_windows[i] = int(windows) 
        np_thresholds[i] = int(threshold)

    return np_windows,np_thresholds



"""
Dynamic Param Function
"""
@nb.njit(cache=True)
def dynamic_params(np_klines: np.array, column_indexes: np.array):

    np_windows = np.full(np_klines.shape[0], np.nan)
    np_thresholds = np.full(np_klines.shape[0], np.nan)

    for i in range(np_klines.shape[0]):
        tide = np_klines[i,column_indexes[-1]]
        # if tide == 1:
        #     window = 24
        #     threshold = 1.5
        # else:
        #     window = 288
        #     threshold = 2
        # if (tide < 5) and (tide > -5):
        #     window = 24
        #     threshold = 1.5
        # else:
        #     window = 288
        #     threshold = 2

        # This yields 2.17B, 0.24TP -------------------------------- BUT WHY DOES TPSL implementation suckier than base strat? 
        # if (tide < 5) and (tide > -5):
        #     window = 60
        #     threshold = 1.5
        # else:
        #     window = 240
        #     threshold = 2

        ## This yields 2.82B, 0.44TP
        if (tide < 5) and (tide > -5):
            window = 240
            threshold = 1.5
        else:
            window = 240
            threshold = 2

        np_windows[i] = int(window)# how to ensure all int? 
        np_thresholds[i] = threshold

    return np_windows,np_thresholds



"""
Main entry point 
"""
def calc_signal_z(df0, 
                    calc_params = static_params,
                    target_cols = ["close"],
                    suffix = "",
                    dynamic_param_col = None,
                    verbose: bool = False
                    ):
    
    df=df0.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index.astype(np.int64) // 10 ** 9
    
    t0 = time.time()
    # Convert dataframe to np.array and get column_indexes for use in param calculations
    if dynamic_param_col is None:
        col_names = ['timestamp'] + target_cols
    else:
        col_names = ['timestamp'] + target_cols + dynamic_param_col

    column_indexes = find_list_index(col_names, dynamic_param_col) 
    column_indexes = np.array(column_indexes) 
    np_klines = df[col_names].values

    t0 = time.time()
    if verbose: print(f"--> Calculating params ...")
    # print(f"np_klines: {np_klines}\ncolumn_indexes: {column_indexes}")
    np_windows, np_thresholds = calc_params(np_klines, column_indexes)
    if verbose: print(f"--< Time taken to calculate params: {np.round(time.time()-t0,2)}s")

    t0 = time.time()
    if verbose: print(f"--> Calculating zscore signals ...")

    np_price = df[target_cols].values
    np_z, np_zsig = rolling_zscore(np_price=np_price,
                          np_windows=np_windows,
                          np_thresholds=np_thresholds, 
                          )
    if verbose: print(f"--< Time taken to calculate zscore signals: {np.round(time.time()-t0,2)}s")
    df[f"z{suffix}"] = np_z
    df[f"zscore{suffix}"] = np_zsig
    df[f"zscore{suffix}"].replace(0, np.nan, inplace=True)
    df[f"zscore{suffix}"].fillna(method="ffill",inplace=True)
    df[f"zscore{suffix}"].fillna(0,inplace=True)
    
    
    return df

