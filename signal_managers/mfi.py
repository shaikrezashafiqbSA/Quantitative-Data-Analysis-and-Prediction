import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm

from utils.find_index import find_list_index
from utils.list_type_converter import convert_to_type,convert_to_array





# ============================================================================================================================================
# ============================================================================================================================================
#                               Base MFI function
# ============================================================================================================================================
# ============================================================================================================================================


@nb.njit(cache=True)
def rolling_mfi(high: np.array, low: np.array, close: np.array, volume: np.array, window: int) -> np.array:
    n = len(close)
    typicalPrice = (high+low+close) / 3
    raw_money_flow = volume * typicalPrice

    # create an index of Bool type
    pos_mf_idx = np.full(n, False)
    pos_mf_idx[1:] = np.diff(typicalPrice) > 0

    # assign values of raw_money_flow to pos_mf where pos_mf_idx == True...Likewise for neg_mf
    pos_mf = np.full(n, np.nan)
    neg_mf = np.full(n, np.nan)
    pos_mf[pos_mf_idx] = raw_money_flow[pos_mf_idx]
    neg_mf[~pos_mf_idx] = raw_money_flow[~pos_mf_idx]

    psum = np.full(n, np.nan)
    nsum = np.full(n, np.nan)

    for i in range(window, n):
        psum[i] = np.nansum(pos_mf[i-window+1:i+1])
        nsum[i] = np.nansum(neg_mf[i-window+1:i+1])

    mfi = 100 * psum / (psum + nsum)

    return mfi


"""
Main entry point 
"""
def calc_signal_mfis(df0,
                     window,
                     label=14
                     ):
    df = df0.copy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    
    mfi = rolling_mfi(high,low, close, volume, window)
    df[f"MFI_{label}"]= mfi
    
    # if ohlc is not None:
    #     df[[f"MFI_{label}_open",f"MFI_{label}_high",f"MFI_{label}_low",f"MFI_{label}_close"]] = calc_ohlc_from_series(df,col_name=f"MFI_{label}", window=ohlc)

    return df


@nb.njit(cache=True)
def rolling_mfi_sharpe(mfi: np.array, window: int) -> np.array:
    n = len(mfi)
    MFI = np.full(n, np.nan)
    for i in range(len(mfi)):
        # Calculate the volatility of the input data
        MFI = mfi[i+1-window:i+1]
        MFI_std = np.std(MFI)
        if MFI_std == 0:
            MFI_sharpe = 0
        else:
            MFI_sharpe = np.mean(MFI)/MFI_std # This should be a EMA vol calculated outside\
        MFI[i] = MFI_sharpe
    return MFI


def calc_signal_mfis_sharpe(df0,feature= "MFI", window=80):
    df = df0.copy()
    mfi = df[feature].to_numpy()
    
    mfi = rolling_mfi_sharpe(mfi, window)
    df[f"{feature}_S_{window}"]= mfi
    
    return df


# ============================================================================================================================================
# ============================================================================================================================================
#                                TO BE GARBAGED. DISGUSTINGLY OVERCOMPLICATED CODE BELOW - DO NOT USE
# ============================================================================================================================================
# ============================================================================================================================================
# # @nb.njit(cache=True)
# def rolling_mfi(np_col: np.array,
#                 #  param_func_mfi,
#                  np_windows: np.array,
#                  fixed_window:bool=False,
#                  fill_value = np.nan,
#                  col_index=None) -> np.array:
#     n = len(np_col)
#     # windows = param_func_mfi(np_col, 0, col_index)
#     print(f"np_windows: {np_windows}")
#     windows = np_windows[0]
#     max_lookback = np.max(windows) # this could be list
#     MFI_i_np = np.full((len(windows)), np.nan)
#     MFI_list = [MFI_i_np]*max_lookback
#     for i in range(max_lookback, n+1):
#         # print(f"i: {i}")
#         # windows = param_func_mfi(np_col, i,col_index)
#         windows = np_windows[i]
#         MFI_i_np = np.full((len(windows)), np.nan)
#         psum = np.full((len(windows)), np.nan)
#         nsum = np.full((len(windows)), np.nan)
#         psum_clean = np.full((len(windows)), np.nan)
#         nsum_clean = np.full((len(windows)), np.nan)
#         for w, window in enumerate(windows):
#             max_lookback = window # this could be list
#             if fixed_window:
#                 high_i = np_col[i-max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
#                 low_i = np_col[i-max_lookback:i+1,3] 
#                 close_i = np_col[i-max_lookback:i+1,4]
#                 volume_i = np_col[i-max_lookback:i+1,-1]
#                 # print(f"i: {i}, maxlookback: {max_lookback} --> high_i: {high_i}")
#             else:
#                 # expanding window
#                 # print(np_col)
#                 high_i = np_col[:i+1,2]
#                 low_i = np_col[:i+1,3]
#                 close_i = np_col[:i+1,4]
#                 volume_i = np_col[:i+1,-1]
#             # print(f"volume_i = np_col[-max_lookback:i+1,-1] : {volume_i} = {len(np_col[-max_lookback:i+1,-1])}")
#             typicalPrice = (high_i+low_i+close_i) / 3
#             # print(f"typicalPrice = (high_i+low_i+close_i) / 3: {typicalPrice} = ({high_i}+{low_i}+{close_i}) / 3")
#             rawMoneyFlow = typicalPrice * volume_i 
#             typicalPrice_diff = np.append(np.nan,np.diff(typicalPrice))
#             # print(f"rawMoney shape: {np.shape(rawMoneyFlow)}")

#             # print(f"rawMoneyFlow = typicalPrice * volume_i: {rawMoneyFlow} = {typicalPrice} * {volume_i}")
#             psum[w] = 0
#             nsum[w] = 0
#             for j in range(int(window)):
#                 # print(f"i: {i} ----> ln 262: i-j-1: {i-j-1}")
#                 typicalPrice_diff_j = typicalPrice_diff[j-1]
#                 if typicalPrice_diff_j <= 0 and not np.isnan(typicalPrice_diff_j):
#                     nsum[w] -= typicalPrice_diff_j#rawMoneyFlow[j-1]
#                 elif typicalPrice_diff_j > 0 and not np.isnan(typicalPrice_diff_j):
#                     psum[w] += typicalPrice_diff_j #rawMoneyFlow[j-1]
#                 else:
#                     psum[w] += 0  #rawMoneyFlow[j-1]
#                     nsum[w] -= 0 #rawMoneyFlow[j-1]
#                 # print(typicalPrice_diff_j)
#             nan_mask = np.isnan(psum[w]) | np.isnan(nsum[w])
#             inf_mask = np.isinf(psum[w]) | np.isinf(nsum[w])

#             # Replace NaN, Inf, and -Inf values with np.nan
#             psum_clean[w] = np.where(nan_mask | inf_mask, np.nan, psum[w])
#             nsum_clean[w] = np.where(nan_mask | inf_mask, np.nan, nsum[w])

#             denominator = psum_clean + nsum_clean
#             if denominator == 0:
#                 mfi = np.nan
#             numerator = 100 * psum_clean[w]
#             mfi = np.where(denominator != 0, numerator/denominator, fill_value)

    
#             MFI_i_np[w] = mfi
#         MFI_list.append(MFI_i_np)

#     return MFI_list


# def param_func_mfi_EMAVol(x, i, col_index = None):
#     q_Lb, q_Ls, q_Sb, q_Ss = 0.8, 0.2, 0.2, 0.8
#     # Calculate the volatility of the input data
#     try:
#         ret_vol = x[:i+1,col_index[0]]
#         vol_sharpe = np.mean(ret_vol)/np.std(ret_vol) # This should be a EMA vol calculated outside\
#         ema_sharpe_spam = vol_sharpe.ewm(span=81)
#         ema_sharpe = ema_sharpe_spam.mean()
#         q_Lb = ema_sharpe_spam.quantile(q=q_Lb)
#         q_Ls = ema_sharpe_spam.quantile(q=q_Ls)
#         q_Sb = ema_sharpe_spam.quantile(q=q_Sb)
#         q_Ss = ema_sharpe_spam.quantile(q=q_Ss)
#         # print(f"EMA sharpe: {ema_sharpe}")
#     except Exception as e:
#         ema_sharpe = 0
#     # print(f"volatility in dynamic_params_func {i}: {volatility}") # This is too smooth to be vol triggers have to change for future use
#     # Set the window lengths and threshold values based on the volatility
#     if 0.67 <= ema_sharpe:
#         windows = [14]
#     elif ema_sharpe < 0.67:
#         windows = [28]
#     else:
#         windows = [55]
    
#     return convert_to_array(windows)

# @nb.njit(cache=True)
# def param_func_mfi(x, i, col_index):
#     """
#     Template for dynamic parameters function
#     where x is a np_array of cols + dynamic_param_col
#     """
#     windows = np.array([24])

#     return windows
    

# def calc_mfi_sig(df0, 
#                  cols_set = [['high','low', 'close', 'volume'], ['high','low', 'open', 'volume']],
#                  param_func_mfi = param_func_mfi,
#                  dynamic_param_col = None,
#                  dynamic_param_combine = True,
#                  fixed_window = False,
#                  col_index = None):  # fixed_window can also be dynamicised away by VARYING expanding window based on volatility
   
#     df = df0.copy() # if somehow need to save initial state of df before adding signals
#     df["date_time"] = df.index
#     n = len(df)
#     # if param_func_mfi is int:
#     np_windows = np.full(n, param_func_mfi)

#     # print(f"n: {n} --> STARTING PARAMETER CALCULATIONS")
#     # for i in tqdm(range(n)):
#     #     np_windows[i] = param_func_mfi(df.values, i, col_index)

#     # print(f"np_windows: {np_windows}")
#     for cols in cols_set:
#         if dynamic_param_col is not None:
#             col_names = ['date_time']+cols+dynamic_param_col
#             col_index = find_list_index(col_names, dynamic_param_col)
#             np_col = df[col_names].values
#         else:
#             col_names = ['date_time']+cols
#             np_col = df[col_names].values


#         # ========================
#         # Calculating MFI
#         # ========================
#         mfi = rolling_mfi(np_col,
#                           np_windows = np_windows,
#                           fixed_window = fixed_window,
#                           col_index = col_index)

#         df1 = df.copy()

#         # ========================
#         # Adding MFI to df
#         # ========================
        
#         for t,i in zip(df1.index, range(len(df1))):
#             # print(f"t: {t}, i: {i}")
#             windows = param_func_mfi(np_col, i, col_index)
#             # print(f"i-{i}: post processing\n--> tide shape: {np.shape(mfi[i])} --> {mfi[i]}\n--> windows: {windows}")
#             for w, window in enumerate(windows):
#                 # print(w)
#                 if dynamic_param_combine:
#                     try:
#                         df1.at[t, f"MFI"] = mfi[i][w]
#                     except Exception:
#                         print(f"shape mfi[i][w]: np.shape(mfi[{i}][{w}]) --> mfi: {mfi}")
#                         df1.at[t, f"MFI"] = -1000 # to find out where in df is error coming from
#                 else:
#                     df1.at[t, f"MFI{window}"] = mfi[i][w]

#         df = df1.copy()
#     return df