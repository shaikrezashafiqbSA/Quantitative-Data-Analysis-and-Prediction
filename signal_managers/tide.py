
import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm
import time

from utils.find_index import find_list_index


# ============================================================================================================================================
# ============================================================================================================================================
#                               Base Tide Functions
# ============================================================================================================================================
# ============================================================================================================================================


@nb.njit(cache=True)
def rolling_sum(heights, w=4):
    ret = np.cumsum(heights)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:]


@nb.njit(cache=True)
def calc_exponential_height(heights, w):  
    rolling_sum_H_L = rolling_sum(heights, w)
    exp_height = (rolling_sum_H_L[-1] - heights[-w] + heights[-1]) / w
    return exp_height 


@nb.njit(cache=True)
def calc_tide(open_i: np.array,
              high_i: np.array,
              low_i: np.array,
              previous_tide: float,
              previous_ebb: float,
              previous_flow: float,
              windows: np.array,
              thresholds: int,
              sensitivity: float) -> np.array:
    
    new_open = open_i[-1]
    new_high = high_i[-1]
    new_low = low_i[-1]

    if np.isnan(previous_tide):
        previous_tide = 1
    if np.isnan(previous_ebb):
        previous_ebb = new_open

    if previous_tide:
        if new_open < previous_ebb:
            undertow = 1
        else:
            undertow = 0
    else:
        if new_open > previous_ebb:
            undertow = 1
        else:
            undertow = 0

    if previous_tide:
        if new_low < previous_ebb:
            surftow = 1
        else:
            surftow = 0
    else:
        if new_high > previous_ebb:
            surftow = 1
        else:
            surftow = 0

    heights = high_i - low_i
    heights = heights[-max(windows):]

    w_0 = 0
    for w in windows:
        w_i = calc_exponential_height(heights, w)
        if w_i > w_0:
            max_exp_height = w_i
            w_0 = w_i

    max_exp_height_ranges = list(np.quantile(heights, np.linspace(0, 1, thresholds)))
    max_exp_height_ranges = [0] + max_exp_height_ranges + [np.inf]
    additives_range = np.linspace(0, np.quantile(heights, sensitivity), len(max_exp_height_ranges) + 1)
    max_exp_height_ranges = list(zip(max_exp_height_ranges[0:-1], max_exp_height_ranges[1:]))

    i = 0
    for maxmpb_range_i in max_exp_height_ranges:
        if maxmpb_range_i[0] <= max_exp_height <= maxmpb_range_i[1]:
            additive = additives_range[i]
            break
        else:
            i += 1

    if previous_tide:
        flow = previous_ebb + additive
    else:
        flow = previous_ebb - additive

    if new_open >= flow:
        tide_1 = 1
    else:
        tide_1 = 0

    if tide_1:
        if new_low < previous_ebb:
            tide_2 = 1
        else:
            tide_2 = 0
    else:
        if new_high > previous_ebb:
            tide_2 = 1
        else:
            tide_2 = 0

    if tide_2:
        if surftow:
            tide_3 = 1
        else:
            tide_3 = 0
    else:
        if surftow:
            if undertow:
                tide_3 = 0
            else:
                tide_3 = 1
        else:
            tide_3 = 0

    if tide_1:
        if new_low >= flow:
            tide_4 = 1
        else:
            tide_4 = 0
    else:
        if new_high > flow:
            tide_4 = 1
        else:
            tide_4 = 0

    " ebb formulation "
    if tide_3 == 1:
        interim_ebb = previous_ebb
    else:
        if tide_1 == 1:
            interim_ebb = new_low - additive
        elif tide_1 == 0:
            interim_ebb = new_high + additive

    " new ebb "
    if tide_1 == 1:
        if interim_ebb <= previous_ebb:
            new_ebb = previous_ebb
        else:
            new_ebb = interim_ebb
    elif tide_1 == 0:
        if interim_ebb < previous_ebb:
            new_ebb = interim_ebb
        else:
            new_ebb = previous_ebb

    " tide formulation "
    if tide_1 == 1 and tide_4 == 1:
        new_tide = 1
    elif tide_1 == 0 and tide_4 == 0:
        new_tide = 0
    else: 
        new_tide = previous_tide

    weights = [0.35, 0.15, 0.15, 0.35] # example weights
    new_tide = (tide_1 * weights[0] + tide_2 * weights[1] + tide_3 * weights[2] + tide_4 * weights[3]) * 20 - 10
    # scale this to 0 and 100
    # now how to scale it between 0 and 1
    # weights = [0.25, 0.75] # example weights
    # new_tide = (tide_1 * weights[0] + tide_4 * weights[1]) * 20 - 10

    return new_tide, new_ebb, flow


@nb.njit(cache=True)
def rolling_tide(open_: np.array,
                 high: np.array,
                 low: np.array,
                 np_windows: np.array,  
                 np_thresholds: np.array,   
                 np_sensitivity: np.array, 
                 fixed_window: bool
                 ):

    n = len(open_)
    np_tide = np.full(n, np.nan)
    np_ebb = np.full(n, np.nan)
    np_flow = np.full(n, np.nan)
    previous_tide = np.nan
    previous_ebb = np.nan
    previous_flow = np.nan

    for i in range(0,n):
        windows = np_windows[i] 
        windows = np.array([int(window) for window in windows])
        threshold = int(np_thresholds[i])
        sensitivity = np_sensitivity[i]
        max_lookback = np.max(windows)

        if max_lookback > i:
            continue

        if fixed_window:
            start_index = int(i+1-max_lookback)
            end_index = int(i+1)
            
            open_i = open_[start_index:end_index]
            high_i = high[start_index:end_index]
            low_i = low[start_index:end_index]
        else:
            end_index = int(i+1)

            open_i = open_[:end_index]
            high_i = high[:end_index]
            low_i = low[:end_index]

        tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
                                          high_i=high_i,
                                          low_i=low_i,
                                          previous_tide=previous_tide,
                                          previous_ebb=previous_ebb,
                                          previous_flow=previous_flow,
                                          windows=windows,
                                          thresholds=threshold,
                                          sensitivity=sensitivity)
            
        previous_tide = tide_i
        previous_ebb = ebb_i
        previous_flow = flow_i
        
        np_tide[i] = tide_i
        np_ebb[i] = ebb_i
        np_flow[i] = flow_i
    
    return np_tide, np_ebb, np_flow

# ============================================================================================================================================
# ============================================================================================================================================
#                               Tide Functions
# ============================================================================================================================================
# ============================================================================================================================================



"""
Static Param Function
"""
@nb.njit(cache=True)
def static_params(np_klines: np.array, column_indexes: np.array):
    windows = [8, 13, 21]
    threshold = 5
    sensitivity = 0.9
    np_windows = np.full((np_klines.shape[0], 3), np.nan)
    np_thresholds = np.full(np_klines.shape[0], np.nan)
    np_sensitivity = np.full(np_klines.shape[0], np.nan)
    for i in range(np_klines.shape[0]):
        np_windows[i] = np.array(windows) # how to ensure all int? 
        np_thresholds[i] = int(threshold)
        np_sensitivity[i] = sensitivity

    return np_windows,np_thresholds,np_sensitivity



"""
Dynamic Param Function
"""
@nb.njit(cache=True)
def dynamic_params(np_klines: np.array, column_indexes: np.array):

    np_windows = np.full((np_klines.shape[0], 3), np.nan)
    np_thresholds = np.full(np_klines.shape[0], np.nan)
    np_sensitivity = np.full(np_klines.shape[0], np.nan)
    max_lookback = 80

    for i in range(np_klines.shape[0]):
        # Calculate the volatility of the input data
        if i < max_lookback:
            windows = [8, 13, 21]
            threshold = 5
            sensitivity = 0.5
            np_windows[i] = np.array(windows)  
            np_thresholds[i] = int(threshold)
            np_sensitivity[i] = sensitivity
            continue
        MFI = np_klines[i+1-max_lookback:i+1,column_indexes[-1]]
        MFI_std = np.std(MFI)
        if MFI_std == 0.0:
            MFI_sharpe = 0
        else:
            MFI_sharpe = np.mean(MFI)/MFI_std # This should be a EMA vol calculated outside\


        if 0.67 <= MFI_sharpe:
            windows = [8,13,21]
            threshold = 5
            sensitivity = 0.5
        elif MFI_sharpe < 0.67:
            windows = [34,45,55]
            threshold = 7
            sensitivity = 0.5
        else:
            windows = [89,121,144]
            threshold = 10
            sensitivity = 0.5

        np_windows[i] = np.array(windows) # how to ensure all int? 20
        np_thresholds[i] = int(threshold) # 0.9
        np_sensitivity[i] = sensitivity # 0.1

    return np_windows,np_thresholds,np_sensitivity



"""
Main entry point 
"""
def calc_signal_tides(df0, 
                      calc_params = static_params,
                      target_cols = ["open","high","low"],
                      fixed_window = True,
                      suffix = "",
                      dynamic_param_col = None,
                      verbose: bool = False,
                      ):

    df = df0.copy() # if somehow need to save initial state of df before adding signals
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
    np_windows, np_thresholds, np_sensitivity = calc_params(np_klines, column_indexes)
    if verbose: print(f"--< Time taken to calculate params: {np.round(time.time()-t0,2)}s")
    # print(f"np_windows: {len(np_windows)}\nnp_thresholds: {len(np_thresholds)}\nnp_sensitivity: {len(np_sensitivity)}")
    # print(f"np_windows: {np_windows}\nnp_thresholds: {np_thresholds}\nnp_sensitivity: {np_sensitivity}")
    open_ = np_klines[:,1]
    high = np_klines[:,2]
    low = np_klines[:,3]

    t0 = time.time()
    if verbose: print(f"--> Calculating tide signals ...")
    tides, ebb, flow = rolling_tide(open_=open_,
                                    high=high,
                                    low=low,
                                    np_windows=np_windows,
                                    np_thresholds=np_thresholds,
                                    np_sensitivity=np_sensitivity,
                                    fixed_window=fixed_window
                                    )
    if verbose: print(f"--< Time taken to calculate tide signals: {np.round(time.time()-t0,2)}s")
    df[f"tide{suffix}"] = tides
    df[f"ebb{suffix}"] = ebb
    df[f"flow{suffix}"] = flow

    
    return df





# ============================================================================================================================================
# ============================================================================================================================================
#                                TO BE GARBAGED. DISGUSTINGLY OVERCOMPLICATED CODE BELOW - DO NOT USE
# ============================================================================================================================================
# ============================================================================================================================================
# @nb.njit(cache=True)
# def rolling_tide(np_klines,
#                  np_params,
#                  fixed_window: bool,
#                  # param_func_tide, # <---- this should not be a function so that can leverage numba
#                  # col_index=None
#                 ):

#     n = len(np_klines)
    
#     previous_tide = np.nan
#     previous_ebb = np.nan
#     previous_flow = np.nan
#     # print(f"col_index = {col_index}, np_col: {np_col}")
#     # windows_sets, thresholds, sensitivities=param_func_tide(np_col,0, col_index)
#     windows_sets, thresholds, sensitivities = [np_params[0, :3]], [np_params[0, 3]], [np_params[0, 4]]
#     tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#     ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#     flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#     print(f"windows_set: {windows_sets} --> type: {type(windows_sets[0])}")
#     max_lookback = int(np.max(windows_sets[0]))
    
#     tide_list = [tide_i_np]*max_lookback
#     ebb_list = [ebb_i_np]*max_lookback
#     flow_list = [flow_i_np]*max_lookback
#     for i in range(max_lookback, n):
#         # windows_sets, thresholds, sensitivities = param_func_tide(np_col,i, col_index)
#         windows_sets, thresholds, sensitivities = [np_params[i, :3]], [np_params[i, 3]], [np_params[i, 4]]
#         tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#         ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#         flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
#         # print(f"shape tide: {np.shape(tide_i_np)}, shape ebb: {np.shape(ebb_i_np)}, shape flow: {np.shape(flow_i_np)}")
#         for w,windows in enumerate(windows_sets):
#             for th,threshold in enumerate(thresholds):
#                 if fixed_window:
#                     open_i = np_klines[i+1-max_lookback:i+1,1]
#                     high_i = np_klines[i+1-max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
#                     low_i = np_klines[i+1-max_lookback:i+1,3]
#                 else:
#                     # expanding window
#                     open_i = np_klines[:i+1,1]
#                     high_i = np_klines[:i+1,2] 
#                     low_i = np_klines[:i+1,3]


#                 previous_tide=tide_list[i-1][th,w]
#                 previous_ebb=ebb_list[i-1][th,w]
#                 previous_flow=flow_list[i-1][th,w]

#                 windows = np.array(windows)
#                 windows = windows.astype(int)
#                 threshold = int(threshold)
#                 sensitivity = float(sensitivities[0])
#                 tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
#                                                 high_i=high_i,
#                                                 low_i=low_i,
#                                                 previous_tide=previous_tide,
#                                                 previous_ebb=previous_ebb,
#                                                 previous_flow=previous_flow,
#                                                 windows=windows,
#                                                 thresholds=threshold,
#                                                 sensitivity=sensitivity)
#                 tide_i_np[th, w] = tide_i
#                 ebb_i_np[th, w] = ebb_i
#                 flow_i_np[th, w] = flow_i

#         tide_list.append(tide_i_np)
#         ebb_list.append(ebb_i_np)
#         flow_list.append(flow_i_np)
        
#     return tide_list,ebb_list,flow_list

# @nb.njit(cache=True)
# def nb_tide(open_: np.array,
#             high: np.array,
#             low: np.array,
#             windows: np.array,
#             thresholds: int,
#             sensitivity: float,
#             fixed_window: bool):

#     n = len(open_)
#     max_lookback = np.max(windows)
#     tide = np.full(n, np.nan)
#     ebb = np.full(n, np.nan)
#     flow = np.full(n, np.nan)
#     previous_tide = np.nan
#     previous_ebb = np.nan
#     previous_flow = np.nan

#     for i in range(max_lookback, n + 1):
#         if fixed_window:
#             open_i = open_[i - max_lookback:i]
#             high_i = high[i - max_lookback:i]
#             low_i = low[i - max_lookback:i]

#         else:

#             open_i = open_[:i]
#             high_i = high[:i]
#             low_i = low[:i]
#         tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
#                                           high_i=high_i,
#                                           low_i=low_i,
#                                           # close_i=close_i,
#                                           previous_tide=previous_tide,
#                                           previous_ebb=previous_ebb,
#                                           previous_flow=previous_flow,
#                                           windows=windows,
#                                           thresholds=thresholds,
#                                           sensitivity=sensitivity)
            
#         previous_tide = tide_i
#         previous_ebb = ebb_i
#         previous_flow = flow_i
        
#         # tide_ids[i-1] = tide_id
#         tide[i - 1] = tide_i
#         ebb[i - 1] = ebb_i
#         flow[i - 1] = flow_i
        
#     return tide, ebb, flow


# @nb.njit
# def ema_sharpe_span_numba(x, span):
#     N = x.shape[0]
#     alpha = 2 / (span + 1)
#     ema_sharpe_span = np.empty(N)
#     ema_sharpe_span[0] = x[0]
#     for i in range(1, N):
#         ema_sharpe_span[i] = alpha * x[i] + (1 - alpha) * ema_sharpe_span[i - 1]
#     return ema_sharpe_span


# @nb.njit
# def calc_params_numba(np_klines: np.array, column_indexes: np.array):

#     q_Lb, q_Ls, q_Sb, q_Ss = 0.9, 0.1, 0.1, 0.9
#     span=81
#     np_params = np.full((np_klines.shape[0], 5), np.nan)

#     for i in range(span, np_klines.shape[0]):
#         # Calculate the volatility of the input data
#         x = np_klines[:i+1,:]

#         MFI = x[:,column_indexes[-1]]
#         vol_sharpe = np.mean(MFI)/np.std(MFI) # This should be a EMA vol calculated outside\
#         ema_sharpe_span = ema_sharpe_span_numba(vol_sharpe, span=span)
#         ema_sharpe = np.mean(ema_sharpe_span)
#         q_Lb = np.quantile(ema_sharpe_span, q=q_Lb)
#         q_Ls = np.quantile(ema_sharpe_span, q=q_Ls)
#         q_Sb = np.quantile(ema_sharpe_span, q=q_Sb)
#         q_Ss = np.quantile(ema_sharpe_span, q=q_Ss)


#         if 0.67 <= ema_sharpe:
#             windows = [8, 13, 21]
#             thresholds = [5]
#             sensitivity = [0.5]
#         elif ema_sharpe < 0.67:
#             windows = [34,45, 55]
#             thresholds = [7]
#             sensitivity = [0.5]
#         else:
#             windows = [89,121,144]
#             thresholds = [10]
#             sensitivity = [0.5]

#         np_params[i] = np.array(int(windows) + thresholds + sensitivity)
#     return np_params

# @nb.njit
# def static_params(np_klines: np.array, column_indexes: np.array):
#     windows = [8, 13, 21]
#     thresholds = [5]
#     sensitivity = [0.5]
#     np_params = np.full((np_klines.shape[0], 5), np.nan)
#     for i in range(np_klines.shape[0]):
#         np_params[i] = np.array(windows + thresholds + sensitivity)
#     return np_params

# def calc_tide_sig(df0, 
#              cols_set = [['open','high','low']],
#              calc_params = calc_params_numba,
#              sig_name_1 = "tide",
#              sig_name_2 = "ebb",
#              sig_name_3 = "flow",
#              fixed_window=False,
#             dynamic_param_col = None,
#             dynamic_param_combine = True):

#     df = df0.copy() # if somehow need to save initial state of df before adding signals
#     df["timestamp"] = df.index.astype(np.int64) // 10 ** 9

#     for cols in cols_set:
#         print(f"cols: {cols}")
#         df[cols] = df[cols].copy().fillna(method='ffill')

#         t0 = time.time()
#         # Convert dataframe to np.array and get column_indexes for use in param calculations
#         if dynamic_param_col is None:
#             col_names = ['timestamp'] + cols
#         else:
#             col_names = ['timestamp'] + cols + dynamic_param_col

#         column_indexes = find_list_index(col_names, dynamic_param_col) 
#         column_indexes = np.array(column_indexes) 
#         np_klines = df[col_names].values
#         len_klines = len(np_klines)
#         print(f"Time taken to convert df to np.array: {np.round(time.time()-t0,2)}s\ncolumn_indexes:{column_indexes}, {type(column_indexes)}")

#         t0 = time.time()
#         print(f"Calculating params")
#         # print(f"np_klines: {np_klines}\ncolumn_indexes: {column_indexes}")
#         np_params = calc_params(np_klines, column_indexes)
#         t1 = time.time()
#         print(f"Time taken to calculate params: {np.round(t1-t0,2)}s")
#          # THIS COULD BE SOURCE OF POTENTIAL POINTER / COPY ERROR 
#         # print(f"cols: {cols}, np_col: {np_col}")
#         tide,ebb,flow = rolling_tide(np_klines,
#                                      np_params,
#                                      fixed_window = fixed_window
#                                      )
#         # print(f"shape tide: {np.shape(tide)}, shape ebb: {np.shape(ebb)}, shape flow: {np.shape(flow)}")
#         df1 = df.copy()
#         # print(f"len df1: {len(df1)}")
#         # if window_threshold_func() 
#         for t,i in zip(df1.index, range(len_klines)):
#             # print(f"t: {t}, i: {i}")
#             windows, thresholds, sensitivities = [np_params[i, :3]], [np_params[i, 3]], [np_params[i, 4]]
#             # print(f"i-{i}: post processing\n--> tide shape: {np.shape(tide[i])} --> {tide[i]}\n--> windows: {windows}\n-->thresholds: {thresholds}\n-->sensitivities: {sensitivities}")
#             for w, window in enumerate(windows):
#                 for th, threshold in enumerate(thresholds):
#                     window_label = "-".join([f"{i}" for i in window])
#                     if dynamic_param_combine:
#                         df1.at[t, f"{sig_name_1}"] = tide[i][th, w]
#                         df1.at[t, f"{sig_name_2}"] = ebb[i][th, w]
#                         df1.at[t, f"{sig_name_3}"] = flow[i][th, w]
#                     else:
#                         df1.at[t, f"{sig_name_1}_w{window_label}t{threshold}"] = tide[i][th, w]
#                         df1.at[t, f"{sig_name_2}_w{window_label}t{threshold}"] = ebb[i][th, w]
#                         df1.at[t, f"{sig_name_3}_w{window_label}t{threshold}"] = flow[i][th, w]

#         df = df1.copy()
#     return df1