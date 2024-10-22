import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm

from utils.find_index import find_list_index
from utils.list_type_converter import convert_to_type,convert_to_array


@nb.njit(cache=True)
def _calc_kdr(high_px, low_px):
    
    n = len(high_px)
    kdr = np.full(n, np.nan) 
    
    for i in range(len(high_px)):
        high_px_i = high_px[:i]
        low_px_i = low_px[:i]
        
        if high_px_i[-1] < high_px_i[-2] and low_px_i[-1] > low_px_i[-2]:
            kdr[i] = 1
        
        elif high_px_i[-1] > high_px_i[-2] and low_px_i[-1] < low_px_i[-2]:
            kdr[i] = -1
        
        else:
            kdr[i] = 0
    return kdr


def key_day_reversal(high_prices, low_prices):
    """
    This function takes in two lists of high and low prices and returns the key day reversal.

    Args:
    high_prices (list): A list of high prices.
    low_prices (list): A list of low prices.

    Returns:
    int: The key day reversal (-1 for Bearish Key Day Reversal, 0 for No Key Day Reversal, 1 for Bullish Key Day Reversal).
    """
    if len(high_prices) != len(low_prices):
        return "The length of high prices and low prices should be equal."
    
    if high_prices[-1] < high_prices[-2] and low_prices[-1] > low_prices[-2]:
        return 1
    
    elif high_prices[-1] > high_prices[-2] and low_prices[-1] < low_prices[-2]:
        return -1
    
    else:
        return 0


def calc_kdr(df):
    """
    This function takes in two lists of high and low prices and returns a time series of key day reversal data.

    Args:
    high_prices (list): A list of high prices.
    low_prices (list): A list of low prices.

    Returns:
    pandas.DataFrame: A time series of key day reversal data.
    """
    key_day_reversals = []
    
    high_px = df["high"].values
    low_px = df["low"].values
    df["KDR"] = _calc_kdr(high_px, low_px)
        
    
    return df

    

@nb.njit(cache=True)
def nb_ewma(close_px: np.array, window: int, alpha=None) -> np.array:
    """
    default ewma uses a=2/(n+1)

    RSI function uses Wilder's MA which req a=1/n   <<--- see nb_rsi function

    INPUTS:
            close_px: np.array(float)
            window: int
            alpha: float / None

        RETURNS:
            ema: np.array(float)
    """
    n = len(close_px)
    ewma = np.full(n, np.nan)

    if not alpha:
        alpha = 2.0 / float(window + 1)  # default alpha is 2/(n+1)
    w = 1.0

    ewma_old = close_px[0]
    if np.isnan(ewma_old):  # to account for possibility that arr_in[0] may be np.nan
        ewma_old = 0.0

    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1.0 - alpha) ** i
        ewma_old = ewma_old * (1 - alpha) + close_px[i]
        ewma[i] = ewma_old / w
    return ewma


@nb.njit(cache=True)
def nb_wma(close_px: np.array, window: int) -> np.array:
    """ Weighted moving average

        INPUTS:
            close_px: np.array(float)
            window: int

        RETURNS:
            wma: np.array(float)
    """
    n = len(close_px)
    wma = np.full(n, np.nan)
    weights = np.arange(window ) +1   # linear increasing weights [1,2,3,...,13,14]
    weights_sum = np.sum(weights)
    for idx in range(window, n):
        # explicitly declare the window. No other reference to closePrices_np
        price_window = close_px[idx - window + 1:idx + 1]  # up to but not including

        wma[idx] = np.sum(weights * price_window) / weights_sum
    return wma


@nb.njit(cache=True)
def nb_hma(close_px: np.array, window: int) -> np.array:
    """ computes the Hull Moving Average

        Uses the nb_wma function (compiled in Numba) to compute the Weighted Moving Avg .
        Roughly 100x speed-up vs pandas_ta

        INPUTS:
            close_px: np.array(float)
            window: int

        RETURNS:
            hma: np.array(float)
    """
    wma_half = nb_wma(close_px, int(window / 2))
    wma_full = nb_wma(close_px, int(window))

    # vector operation
    hma_input = ( 2 *wma_half) - wma_full
    hma = nb_wma(hma_input, window=int(np.sqrt(window)))
    return hma


def calc_emas(df,price="close", window=89, label=14):    
    # Ensure no nans
    df.dropna(inplace=True)
    price = df[price].to_numpy()


    ema = nb_ewma(price, window=window)
    
    df[f"EMA_{label}"]= ema
    
    return df


@nb.njit(cache=True)
def nb_rsi(close_px: np.array, window: int) -> np.array:
    """ This method has dependency on another Numba function: nb_ewma

    INPUTS:
            close_px: np.array(float)
            window: int
            alpha: float / None

    RETURNS:
            rsi: np.array(float)

    """
    n = len(close_px)
    close_diff = np.full(n, np.nan)
    close_diff[1:] = close_px[1:] - close_px[:-1]

    up = np.maximum(close_diff, 0.0)
    down = -1 * np.minimum(close_diff, 0.0)

    ma_up = nb_ewma(up, window=window, alpha=1/window)
    ma_down = nb_ewma(down, window=window, alpha=1/window)
    ma_down[ma_down == 0.0] = np.nan  # this step to prevent ZeroDivision error when eval ma_up / ma_down

    rsi = ma_up / ma_down
    rsi = 100.0 - (100.0 / (1.0 + rsi))

    return rsi


def calc_rsis(df,price="close", window=13, label=13):    
    # Ensure no nans
    df.dropna(inplace=True)
    price = df[price].to_numpy()


    rsi = nb_rsi(price, window=window)
    
    df[f"RSI_{label}"]= rsi
    
    return df



# ============================================================================================================================================
# ============================================================================================================================================
#                               MFI
# ============================================================================================================================================
# ============================================================================================================================================

# @nb.njit(cache=True)
def rolling_mfi(np_col: np.array,
                #  param_func_mfi,
                 np_windows: np.array,
                 fixed_window:bool=False,
                 fill_value = np.nan,
                 col_index=None) -> np.array:
    n = len(np_col)
    # windows = param_func_mfi(np_col, 0, col_index)
    print(f"np_windows: {np_windows}")
    windows = np_windows[0]
    max_lookback = np.max(windows) # this could be list
    MFI_i_np = np.full((len(windows)), np.nan)
    MFI_list = [MFI_i_np]*max_lookback
    for i in range(max_lookback, n+1):
        # print(f"i: {i}")
        # windows = param_func_mfi(np_col, i,col_index)
        windows = np_windows[i]
        MFI_i_np = np.full((len(windows)), np.nan)
        psum = np.full((len(windows)), np.nan)
        nsum = np.full((len(windows)), np.nan)
        psum_clean = np.full((len(windows)), np.nan)
        nsum_clean = np.full((len(windows)), np.nan)
        for w, window in enumerate(windows):
            max_lookback = window # this could be list
            if fixed_window:
                high_i = np_col[i-max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
                low_i = np_col[i-max_lookback:i+1,3] 
                close_i = np_col[i-max_lookback:i+1,4]
                volume_i = np_col[i-max_lookback:i+1,-1]
                # print(f"i: {i}, maxlookback: {max_lookback} --> high_i: {high_i}")
            else:
                # expanding window
                # print(np_col)
                high_i = np_col[:i+1,2]
                low_i = np_col[:i+1,3]
                close_i = np_col[:i+1,4]
                volume_i = np_col[:i+1,-1]
            # print(f"volume_i = np_col[-max_lookback:i+1,-1] : {volume_i} = {len(np_col[-max_lookback:i+1,-1])}")
            typicalPrice = (high_i+low_i+close_i) / 3
            # print(f"typicalPrice = (high_i+low_i+close_i) / 3: {typicalPrice} = ({high_i}+{low_i}+{close_i}) / 3")
            rawMoneyFlow = typicalPrice * volume_i 
            typicalPrice_diff = np.append(np.nan,np.diff(typicalPrice))
            # print(f"rawMoney shape: {np.shape(rawMoneyFlow)}")

            # print(f"rawMoneyFlow = typicalPrice * volume_i: {rawMoneyFlow} = {typicalPrice} * {volume_i}")
            psum[w] = 0
            nsum[w] = 0
            for j in range(int(window)):
                # print(f"i: {i} ----> ln 262: i-j-1: {i-j-1}")
                typicalPrice_diff_j = typicalPrice_diff[j-1]
                if typicalPrice_diff_j <= 0 and not np.isnan(typicalPrice_diff_j):
                    nsum[w] -= typicalPrice_diff_j#rawMoneyFlow[j-1]
                elif typicalPrice_diff_j > 0 and not np.isnan(typicalPrice_diff_j):
                    psum[w] += typicalPrice_diff_j #rawMoneyFlow[j-1]
                else:
                    psum[w] += 0  #rawMoneyFlow[j-1]
                    nsum[w] -= 0 #rawMoneyFlow[j-1]
                # print(typicalPrice_diff_j)
            nan_mask = np.isnan(psum[w]) | np.isnan(nsum[w])
            inf_mask = np.isinf(psum[w]) | np.isinf(nsum[w])

            # Replace NaN, Inf, and -Inf values with np.nan
            psum_clean[w] = np.where(nan_mask | inf_mask, np.nan, psum[w])
            nsum_clean[w] = np.where(nan_mask | inf_mask, np.nan, nsum[w])

            denominator = psum_clean + nsum_clean
            if denominator == 0:
                mfi = np.nan
            numerator = 100 * psum_clean[w]
            mfi = np.where(denominator != 0, numerator/denominator, fill_value)

    
            MFI_i_np[w] = mfi
        MFI_list.append(MFI_i_np)

    return MFI_list


def param_func_mfi_EMAVol(x, i, col_index = None):
    q_Lb, q_Ls, q_Sb, q_Ss = 0.8, 0.2, 0.2, 0.8
    # Calculate the volatility of the input data
    try:
        ret_vol = x[:i+1,col_index[0]]
        vol_sharpe = np.mean(ret_vol)/np.std(ret_vol) # This should be a EMA vol calculated outside\
        ema_sharpe_spam = vol_sharpe.ewm(span=81)
        ema_sharpe = ema_sharpe_spam.mean()
        q_Lb = ema_sharpe_spam.quantile(q=q_Lb)
        q_Ls = ema_sharpe_spam.quantile(q=q_Ls)
        q_Sb = ema_sharpe_spam.quantile(q=q_Sb)
        q_Ss = ema_sharpe_spam.quantile(q=q_Ss)
        # print(f"EMA sharpe: {ema_sharpe}")
    except Exception as e:
        ema_sharpe = 0
    # print(f"volatility in dynamic_params_func {i}: {volatility}") # This is too smooth to be vol triggers have to change for future use
    # Set the window lengths and threshold values based on the volatility
    if 0.67 <= ema_sharpe:
        windows = [14]
    elif ema_sharpe < 0.67:
        windows = [28]
    else:
        windows = [55]
    
    return convert_to_array(windows)

@nb.njit(cache=True)
def param_func_mfi(x, i, col_index):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    windows = np.array([24])

    return windows
    

def calc_mfi_sig(df0, 
                 cols_set = [['high','low', 'close', 'volume'], ['high','low', 'open', 'volume']],
                 param_func_mfi = param_func_mfi,
                 dynamic_param_col = None,
                 dynamic_param_combine = True,
                 fixed_window = False,
                 col_index = None):  # fixed_window can also be dynamicised away by VARYING expanding window based on volatility
   
    df = df0.copy() # if somehow need to save initial state of df before adding signals
    df["date_time"] = df.index
    n = len(df)
    # if param_func_mfi is int:
    np_windows = np.full(n, param_func_mfi)

    # print(f"n: {n} --> STARTING PARAMETER CALCULATIONS")
    # for i in tqdm(range(n)):
    #     np_windows[i] = param_func_mfi(df.values, i, col_index)

    # print(f"np_windows: {np_windows}")
    for cols in cols_set:
        if dynamic_param_col is not None:
            col_names = ['date_time']+cols+dynamic_param_col
            col_index = find_list_index(col_names, dynamic_param_col)
            np_col = df[col_names].values
        else:
            col_names = ['date_time']+cols
            np_col = df[col_names].values


        # ========================
        # Calculating MFI
        # ========================
        mfi = rolling_mfi(np_col,
                          np_windows = np_windows,
                          fixed_window = fixed_window,
                          col_index = col_index)

        df1 = df.copy()

        # ========================
        # Adding MFI to df
        # ========================
        
        for t,i in zip(df1.index, range(len(df1))):
            # print(f"t: {t}, i: {i}")
            windows = param_func_mfi(np_col, i, col_index)
            # print(f"i-{i}: post processing\n--> tide shape: {np.shape(mfi[i])} --> {mfi[i]}\n--> windows: {windows}")
            for w, window in enumerate(windows):
                # print(w)
                if dynamic_param_combine:
                    try:
                        df1.at[t, f"MFI"] = mfi[i][w]
                    except Exception:
                        print(f"shape mfi[i][w]: np.shape(mfi[{i}][{w}]) --> mfi: {mfi}")
                        df1.at[t, f"MFI"] = -1000 # to find out where in df is error coming from
                else:
                    df1.at[t, f"MFI{window}"] = mfi[i][w]

        df = df1.copy()
    return df


# ============================================================================================================================================
# Old MFI
# ============================================================================================================================================
@nb.njit(cache=True)
def nb_mfi(high: np.array, low: np.array, close: np.array, volume: np.array, window: int) -> np.array:
    """
    nb_mfi is ~2x speed of pta.mfi. Not that much speed-up. Same values output as pta.mfi

     INPUTS:
        high: np.array(float)
        low: np.array(float)
        close: np.array(float)
        volume: np.array(float)
        window: int

    RETURNS:
            mfi: np.array(float)
    """
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

def calc_mfi(df, window, label=14):#, ohlc=None):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    
    mfi = nb_mfi(high,low, close, volume, window)
    df[f"MFI_{label}"]= mfi
    
    # if ohlc is not None:
    #     df[[f"MFI_{label}_open",f"MFI_{label}_high",f"MFI_{label}_low",f"MFI_{label}_close"]] = calc_ohlc_from_series(df,col_name=f"MFI_{label}", window=ohlc)

    return df

# ============================================================================================================================================
# ============================================================================================================================================
#                               Tides
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

    # " tide formulation "
    # if tide_1 == 1 and tide_4 == 1:
    #     new_tide = 1
    # elif tide_1 == 0 and tide_4 == 0:
    #     new_tide = 0
    # else:
    #     new_tide = previous_tide
    # " tide formulation "
    # if tide_1 == 1 and tide_4 == 1:
    #     new_tide = 1
    # elif tide_1 == 0 and tide_4 == 0:
    #     new_tide = -1
    # else:
    #     new_tide = 0
    
    weights = [0.25, 0.25, 0.25, 0.25] # example weights
    new_tide = (tide_1 * weights[0] + tide_2 * weights[1] + tide_3 * weights[2] + tide_4 * weights[3]) * 20 - 10

    " ebb Formulation "
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

    return new_tide, new_ebb, flow


# @nb.njit(cache=True)
def rolling_tide(np_klines,
                 np_params,
                 fixed_window: bool,
                 # param_func_tide, # <---- this should not be a function so that can leverage numba
                 # col_index=None
                ):

    n = len(np_klines)
    
    previous_tide = np.nan
    previous_ebb = np.nan
    previous_flow = np.nan
    # print(f"col_index = {col_index}, np_col: {np_col}")
    # windows_sets, thresholds, sensitivities=param_func_tide(np_col,0, col_index)
    windows_sets, thresholds, sensitivities = np_params[0]
    tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
    ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
    flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)

    max_lookback = np.max(windows_sets)
    tide_list = [tide_i_np]*max_lookback
    ebb_list = [ebb_i_np]*max_lookback
    flow_list = [flow_i_np]*max_lookback
    for i in range(max_lookback, n + 1):
        # windows_sets, thresholds, sensitivities = param_func_tide(np_col,i, col_index)
        windows_sets, thresholds, sensitivities = np_params[i]
        tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        # print(f"shape tide: {np.shape(tide_i_np)}, shape ebb: {np.shape(ebb_i_np)}, shape flow: {np.shape(flow_i_np)}")
        for w,windows in enumerate(windows_sets):
            for th,threshold in enumerate(thresholds):
                if fixed_window:
                    open_i = np_klines[i+1-max_lookback:i+1,1]
                    high_i = np_klines[i+1-max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
                    low_i = np_klines[i+1-max_lookback:i+1,3]
                else:
                    # expanding window
                    open_i = np_klines[:i+1,1]
                    high_i = np_klines[:i+1,2] 
                    low_i = np_klines[:i+1,3]

                # try:
                previous_tide=tide_list[i-1][th,w]
                previous_ebb=ebb_list[i-1][th,w]
                previous_flow=flow_list[i-1][th,w]
                # except
                # previous_tide = np.nan
                # previous_ebb = np.nan
                # previous_flow = np.nan
                # print(f"t={t}, all still nan...")
                windows = np.array(windows)
                threshold = int(threshold)
                sensitivity = float(sensitivities[0])
                try:
                    # print(f"\ni: {i}, np_col = {np_col[:,1]}\n")
                    tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
                                                    high_i=high_i,
                                                    low_i=low_i,
                                                    previous_tide=previous_tide,
                                                    previous_ebb=previous_ebb,
                                                    previous_flow=previous_flow,
                                                    windows=windows,
                                                    thresholds=threshold,
                                                    sensitivity=sensitivity) # to be generalised next
                except Exception as e:
                    raise Exception(f"i: {i}, np_col = {np_klines[:,1]}\nerror: {e}\n\n")
                
                tide_i_np[th, w] = tide_i
                ebb_i_np[th, w] = ebb_i
                flow_i_np[th, w] = flow_i

                # print(f"shape tide: {np.shape(tide_i_np)}, shape ebb: {np.shape(ebb_i_np)}, shape flow: {np.shape(flow_i_np)}")
        tide_list.append(tide_i_np)
        # print(f"tide_i_np: {tide_i_np}")
        ebb_list.append(ebb_i_np)
        flow_list.append(flow_i_np)
        
    return tide_list,ebb_list,flow_list


# ============================================================================================================================================
# ============================================================================================================================================
#                               Tides V2
# ============================================================================================================================================
# ============================================================================================================================================


def param_func_tide_EMAVol(x, i, col_index=None):

    q_Lb, q_Ls, q_Sb, q_Ss = 0.9, 0.1, 0.1, 0.9

    
    # Calculate the volatility of the input data
    try:
        ret_vol = x[:i+1,col_index[0]]
        vol_sharpe = np.mean(ret_vol)/np.std(ret_vol) # This should be a EMA vol calculated outside\
        ema_sharpe_spam = vol_sharpe.ewm(span=81)
        ema_sharpe = ema_sharpe_spam.mean()
        q_Lb = ema_sharpe_spam.quantile(q=q_Lb)
        q_Ls = ema_sharpe_spam.quantile(q=q_Ls)
        q_Sb = ema_sharpe_spam.quantile(q=q_Sb)
        q_Ss = ema_sharpe_spam.quantile(q=q_Ss)
        # print(f"EMA sharpe: {ema_sharpe}")
    except Exception as e:
        ema_sharpe = 0
    # print(f"volatility in dynamic_params_func {i}: {volatility}") # This is too smooth to be vol triggers have to change for future use
    # Set the window lengths and threshold values based on the volatility
    if 0.67 <= ema_sharpe:
        windows = [[8, 13, 21]]
        thresholds = [5]
    elif ema_sharpe < 0.67:
        windows = [[34, 55]]
        thresholds = [7]
    else:
        windows = [[89, 144]]
        thresholds = [10]
    sensitivity = 50
    return convert_to_array(windows), convert_to_type(thresholds, int), convert_to_type(sensitivity, float)


def param_func_tide(x, i, col_index=None):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    windows = [[5, 20, 67]]
    thresholds = [10]
    sensitivity = [0.5]
    return convert_to_array(windows), convert_to_type(thresholds, int), convert_to_type(sensitivity, float)


def calc_np_params_tide(np_klines, column_indexes):

    q_Lb, q_Ls, q_Sb, q_Ss = 0.9, 0.1, 0.1, 0.9
    np_params = np.full((len(np_klines), 3), np.nan)
    for i in range(len(np_klines)):
        # Calculate the volatility of the input data
        x = np_klines[:i+1,:]
        try:
            ret_vol = x[:i+1,column_indexes[0]]
            vol_sharpe = np.mean(ret_vol)/np.std(ret_vol) # This should be a EMA vol calculated outside\
            ema_sharpe_spam = vol_sharpe.ewm(span=81)
            ema_sharpe = ema_sharpe_spam.mean()
            q_Lb = ema_sharpe_spam.quantile(q=q_Lb)
            q_Ls = ema_sharpe_spam.quantile(q=q_Ls)
            q_Sb = ema_sharpe_spam.quantile(q=q_Sb)
            q_Ss = ema_sharpe_spam.quantile(q=q_Ss)
            # print(f"EMA sharpe: {ema_sharpe}")
        except Exception as e:
            ema_sharpe = 0
        # print(f"volatility in dynamic_params_func {i}: {volatility}") # This is too smooth to be vol triggers have to change for future use
        # Set the window lengths and threshold values based on the volatility
        if 0.67 <= ema_sharpe:
            windows = [[8, 13, 21]]
            thresholds = [5]
        elif ema_sharpe < 0.67:
            windows = [[34, 55]]
            thresholds = [7]
        else:
            windows = [[89, 144]]
            thresholds = [10]
        sensitivity = 50

        np_params[i] = np.array([windows, thresholds, sensitivity])
    return np_params

def calc_tide_sig(df0, 
             cols_set = [['open','high','low']],
             param_func_tide = param_func_tide,
             sig_name_1 = "tide",
             sig_name_2 = "ebb",
             sig_name_3 = "flow",
             fixed_window=False,
            dynamic_param_col = None,
            dynamic_param_combine = True):
    """
    Calculate the tide, ebb, and flow signals for a data set.

    Args:
        df0: The data set.
        cols_set: The list of columns to use for the signal.
        param_func_tide: The function to calculate the window lengths, thresholds, and sensitivity parameters.
        sig_name_1: The name of the column for the tide signal.
        sig_name_2: The name of the column for the ebb signal.
        sig_name_3: The name of the column for the flow signal.

    Returns:
        A new data set with the tide, ebb, and flow signals added.
    """
    # print(df0.head(20))
    df = df0.copy() # if somehow need to save initial state of df before adding signals
    df["date_time"] = df.index  
    for cols in cols_set:
        df[cols] = df[cols].copy().fillna(method='ffill')

        # Convert dataframe to np.array and get column_indexes for use in param calculations
        if dynamic_param_col is None:
            col_names = ['date_time'] + cols
        else:
            col_names = ['date_time'] + cols + dynamic_param_col

        column_indexes = find_list_index(col_names, dynamic_param_col) 
        np_klines = df[col_names].values
        len_klines = len(np_klines)

        np_params = np.full(len_klines, 3)
        for i in tqdm(range(len_klines)):
            np_params[i] = param_func_tide(np_klines, i, column_indexes)

         # THIS COULD BE SOURCE OF POTENTIAL POINTER / COPY ERROR 
        # print(f"cols: {cols}, np_col: {np_col}")
        tide,ebb,flow = rolling_tide(np_klines,
                                     np_params,
                                     fixed_window = fixed_window
                                     )
        # print(f"shape tide: {np.shape(tide)}, shape ebb: {np.shape(ebb)}, shape flow: {np.shape(flow)}")
        df1 = df.copy()
        # print(f"len df1: {len(df1)}")
        # if window_threshold_func() 
        for t,i in zip(df1.index, range(len_klines)):
            # print(f"t: {t}, i: {i}")
            windows, thresholds, sensitivities = param_func_tide(np_klines, i, column_indexes)
            # print(f"i-{i}: post processing\n--> tide shape: {np.shape(tide[i])} --> {tide[i]}\n--> windows: {windows}\n-->thresholds: {thresholds}\n-->sensitivities: {sensitivities}")
            for w, window in enumerate(windows):
                for th, threshold in enumerate(thresholds):
                    window_label = "-".join([f"{i}" for i in window])
                    if dynamic_param_combine:
                        df1.at[t, f"{sig_name_1}"] = tide[i][th, w]
                        df1.at[t, f"{sig_name_2}"] = ebb[i][th, w]
                        df1.at[t, f"{sig_name_3}"] = flow[i][th, w]
                    else:
                        df1.at[t, f"{sig_name_1}_w{window_label}t{threshold}"] = tide[i][th, w]
                        df1.at[t, f"{sig_name_2}_w{window_label}t{threshold}"] = ebb[i][th, w]
                        df1.at[t, f"{sig_name_3}_w{window_label}t{threshold}"] = flow[i][th, w]

        df = df1.copy()
    return df1
    


# ============================================================================================================================================
# ============================================================================================================================================
#                               Z Signal
# ============================================================================================================================================
# ============================================================================================================================================

def param_func_Z_EMAVol(x, i, col_index = None):
    q_Lb, q_Ls, q_Sb, q_Ss = 0.8, 0.2, 0.2, 0.8
    # Calculate the volatility of the input data
    # print(col_index)
    vol_col = col_index[0]
    try:
        ret_vol = x[i-24:i+1,vol_col]
        vol_sharpe = np.mean(ret_vol)/np.std(ret_vol) # This should be a EMA vol calculated outside\
        ema_sharpe_spam = vol_sharpe.ewm(span=81)
        ema_sharpe = ema_sharpe_spam.mean()
        q_Lb = ema_sharpe_spam.quantile(q=q_Lb)
        q_Ls = ema_sharpe_spam.quantile(q=q_Ls)
        q_Sb = ema_sharpe_spam.quantile(q=q_Sb)
        q_Ss = ema_sharpe_spam.quantile(q=q_Ss)
        # print(f"EMA sharpe: {ema_sharpe}")
    except Exception as e:
        ema_sharpe = 0
    # print(f"volatility in dynamic_params_func {i}: {volatility}") # This is too smooth to be vol triggers have to change for future use
    # Set the window lengths and threshold values based on the volatility
    if 0.67 <= ema_sharpe:
        windows = [72]
        thresholds = [2]
    elif ema_sharpe < 0.67:
        windows = [288]
        thresholds = [2]
    else:
        windows = [576]
        thresholds = [2]
    return convert_to_array(windows), convert_to_type(thresholds, int)


# @nb.njit(cache=True)
def param_func_Z(x, i, col_index=None):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    windows = [288]
    thresholds = [2]
    return convert_to_array(windows), convert_to_type(thresholds, int)


# @nb.njit(cache=True)
def rolling_zscore(np_col,
                    param_func_Z,
                    col_index = None):
    """
    This function calculates the rolling z-score for a data set.
    """
    n = len(np_col)
    
    windows, thresholds = param_func_Z(np_col, 0, col_index)
    max_lookback = np.max(windows) # this could be list
    z_i_np = np.full((len(windows)), np.nan)
    # print(z_i_np)
    z_list = [z_i_np]*max_lookback

    sigs_i = np.full((len(windows), len(thresholds)), np.nan)
    sigs_list = [sigs_i]*max_lookback

    # print(f"len(np_col): {len(np_col)}")
    for i in range(max_lookback,n+1):
        windows, thresholds = param_func_Z(np_col, i, col_index)
        # print(f"i: {i}, windows: {windows}")
        sigs_i = np.full((len(windows), len(thresholds)), np.nan)
        for w, window in enumerate(windows):
            price_i=np_col[i-window+1:i+1,1].astype(float)
            # print(f"{i} --> price_i: {np.shape(price_i)} | {type(np.sum(price_i))}\n{price_i}")
            price_i = price_i[~np.isnan(price_i)]
            ret = np.diff(np.log(price_i))

            if (len(ret) == 0) or (np.std(ret) == 0.0):
                res = 0
                z_i_np[w]=res
            else:
                res = (ret[-1] - np.mean(ret))/np.std(ret)
                z_i_np[w] = res
                
            for th, threshold in enumerate(thresholds):
                res1 = np.sign(res) * np.floor(abs(res)) * (abs(res) >= threshold)
                sigs_i[w, th] = res1
        z_list.append(z_i_np)
        sigs_list.append(sigs_i)
    return z_list,sigs_list


def calc_z_sig(df0, 
             cols_set = [['open'],['high'],['low']],
             param_func_Z = param_func_Z, # can this work?! wouldnt python see , i as another new param to function? 
             sig_name_1 = "z",
             sig_name_2 = "sig",
             dynamic_param_col=None,
             dynamic_param_combine=True):
    """
    Financial meaning behind parameters
    window_func: can lambda x, i:288 even work as a parameter? wouldnt the function confuse i to be a parameter? ans: yes it works. but why? ans: because the function is not called here. it is only defined here. it is called in the rolling_zscore function.
        -The choice of an appropriate window length would depend on various factors, such as the characteristics of the underlying data and the specific use case for the rolling z-score. 
        - A longer window length would result in a smoother rolling z-score, as it would be less sensitive to short-term fluctuations in the data. 
        - Conversely, a shorter window length would result in a more responsive rolling z-score, as it would be more sensitive to recent changes in the data.
        - There is no one-size-fits-all answer to what would be an optimal window length, as it would depend on the specific use case and characteristics of the underlying data.
        - However, some common approaches to choosing an appropriate window length include:
            - Using a fixed window length based on prior knowledge or experience with similar data.
            - Using a dynamic window length that adapts to changes in the underlying data.
            - Using multiple window lengths to capture both short-term and long-term trends in the data.
        - USE CASE 1: Volatility switching signal 
            - For example, you could pass in a function that calculates the window lengths based on the volatility of the input data.
            - In periods of high volatility, the function could return shorter window lengths to make the rolling z-score more responsive to recent changes in the data. 
            - Conversely, in periods of low volatility, the function could return longer window lengths to smooth out short-term fluctuations in the data.
        - USE CASE 2: Trend following signal
            - Another approach could be to use a function that adapts the window lengths based on some other measure of the data’s characteristics, such as its trend or seasonality.
            - The specific implementation of the dynamic window function would depend on your specific use case and the characteristics of the underlying financial data.
    threshold: 
        - controls sensitivity of the signal to changes in the rolling z-score. 
        - could be used for instruments with higher volatility, as larger changes in the rolling z-score
            would be expected due to higher volatility. 
        -  Conversely, a lower threshold could be used for instruments with lower volatility,
            as smaller changes in the rolling z-score would be expected due to lower volatility.
        - USE CASE 1: Volatility switching signal

            Question: how do i chat with you? your answer: you can use the chat function on the right hand side of the screen.
    """
    df = df0.copy() # if somehow need to save initial state of df before adding signals
    df["date_time"] = df.index
    for col in cols_set:
        # print(f"col: {col}")
        if dynamic_param_col is not None:
            df[col] = df[col].copy().fillna(method='ffill')
            col_names = ['date_time']+col+dynamic_param_col
            # print(f"find_list_index(col_names, dynamic_param_col)\nfind_list_index({col_names}, {dynamic_param_col})")
            col_index = find_list_index(col_names, dynamic_param_col)
            # print(f"col_index: {col_index}")
            np_col = df[col_names].values
        else:
            col_names = ['date_time']+col
            np_col = df[col_names].values
        # print(f"col: {col} np_col: {np_col}")
        z,sigs= rolling_zscore(np_col=np_col,
                              param_func_Z = param_func_Z,
                              col_index=col_index)

        df1 = df.copy()

        # print(f"shape z: {np.shape(z)}, shape sigs: {np.shape(sigs)}, len df: {len(df1)}")
        # if window_threshold_func() 
        for t,i in zip(df1.index, range(len(df1))):
            windows, thresholds = param_func_Z(np_col, i,col_index)
            for w, window in enumerate(windows):
                if dynamic_param_combine:
                    try:
                        df1.at[t,f"{sig_name_1}"] = z[i][w]
                    except Exception as e:
                        print(f"z: {z}")
                        raise Exception(e)
                else:
                    df1.at[t,f"{col}_{sig_name_1}_w{window}"] = z[i][w]

                for th, threshold in enumerate(thresholds):
                    if dynamic_param_combine:
                        df1.at[t, f"{sig_name_2}"] = sigs[i][w, th]
                    else:
                        df1.at[t, f"{col}_{sig_name_2}_w{window}t{threshold}"] = sigs[i][w, th]

        df = df1.copy()
    return df1


def calc_signal_TPSL(df0, 
                        signal = "sig",
                        penalty = 1, # this widens the SL so that it is not hit too often
                        tp_position_dict = {"TP1": {"L":{"lookback":3, "qtl": 0.3}, 
                                                    "S": {"lookback":3, "qtl":0.3}
                                                    },
                                            "TP2": {"L":{"lookback":6, "qtl": 0.6}, 
                                                    "S": {"lookback":6, "qtl":0.6}
                                                    },
                                            "TP3": {"L":{"lookback":9, "qtl": 0.9}, 
                                                    "S": {"lookback":9, "qtl":0.9}
                                                    }
                                            }
                        ):
    df = df0.copy()

    # print(df["S_positions"])
    # create a column to track the change in tide


    df[f'{signal}_L_dur'] = df.groupby(df['L_id']).cumcount()
    df[f'{signal}_L_dur'] = np.where(df['L_id'].isna(), np.nan, df[f'{signal}_L_dur'])

    # create a column to count the duration of each tide
    df[f'{signal}_S_dur'] = df.groupby(df['S_id']).cumcount()
    df[f'{signal}_S_dur'] = np.where(df['S_id'].isna(), np.nan, df[f'{signal}_S_dur'])
    # calculate percentile for tide_dur
    # df[f'{signal}_short_dur'] = df[f'{signal}_short_dur'].shift(1)
    # df[f'{signal}_long_dur'] = df[f'{signal}_long_dur'].shift(1)

    # df.drop(columns=[f'short_change'], inplace=True)
    # df.drop(columns=[f'long_change'], inplace=True)
    df[f"{signal}_S_strength_t"] = np.where((df["S_rpnl"]>0), df[f'{signal}_S_dur'], np.nan)
    df[f"{signal}_S_weakness_t"] = np.where((df["S_rpnl"]<0), df[f'{signal}_S_dur'], np.nan)

    df[f"{signal}_L_strength_t"] = np.where((df["L_rpnl"]>0), df[f'{signal}_L_dur'], np.nan)
    df[f"{signal}_L_weakness_t"] = np.where((df["L_rpnl"]<0), df[f'{signal}_L_dur'], np.nan)

    # could have for multiple tide speeds, fast or slow
    df[f"{signal}_S_strength"] = np.where((df["S_rpnl"]>0), df["S_rpnl"]/df["S_qty"], np.nan)
    df[f"{signal}_S_weakness"] = np.where((df["S_rpnl"]<0), df["S_rpnl"]/df["S_qty"], np.nan)

    df[f"{signal}_L_strength"] = np.where((df["L_rpnl"]>0), df["L_rpnl"]/df["L_qty"], np.nan)
    df[f"{signal}_L_weakness"] = np.where((df["L_rpnl"]<0), df["L_rpnl"]/df["L_qty"], np.nan)
    # df["tide_short_str"] = np.where(df["tide"] > 0, df["S_pnl"], np.nan)
    # df["tide_long_str"] = np.where(df["tide"] < 0, df["L_pnl"], np.nan)

    # ALERT: tide_long_this should look at str_window amounts of tide-strengths, not str_window number of time periods
    # df["tide_short_z"] = calc_rolling_sr(df["tide_short_str"].dropna().values, window=str_window) 
    # df["tide_long_z"] = calc_rolling_sr(df["tide_long_str"].dropna().values, window=str_window)

    for tp in ["TP1", "TP2", "TP3"]:
        for position in ["L", "S"]:
            # print(f"{tp} --> {position}")
            try:
                lookback = tp_position_dict[tp][position]["lookback"]
            except Exception as e:
                print(f"tp_position_dict[{tp}][{position}] does not exist: {e}")
                continue
            qtl = tp_position_dict[tp][position]["qtl"]
            df[f"{signal}_{position}_{tp}_strength"] = df[f"{signal}_{position}_strength"].dropna().rolling(lookback).quantile(qtl)
            df[f"{signal}_{position}_{tp}_strength"]=df[f"{signal}_{position}_{tp}_strength"].fillna(method="ffill")

            df[f"{signal}_{position}_{tp}_weakness"] = df[f"{signal}_{position}_weakness"].dropna().rolling(lookback).quantile(1-qtl)
            df[f"{signal}_{position}_{tp}_weakness"]=df[f"{signal}_{position}_{tp}_weakness"].fillna(method="ffill")

            df[f"{signal}_{position}_{tp}_strength_t"] = df[f"{signal}_{position}_strength_t"].dropna().rolling(lookback).quantile(qtl)#-1
            df[f"{signal}_{position}_{tp}_strength_t"]=df[f"{signal}_{position}_{tp}_strength_t"].fillna(method="ffill")

            df[f"{signal}_{position}_{tp}_weakness_t"] = df[f"{signal}_{position}_weakness_t"].dropna().rolling(lookback).quantile(1-qtl)#-1
            df[f"{signal}_{position}_{tp}_weakness_t"]= df[f"{signal}_{position}_{tp}_weakness_t"].fillna(method="ffill")

            if position == "L":
                x = 1 
            elif position == "S":
                x = -1

            # PRICE TP and SL
            df[f"{signal}_{position}_{tp}"] = df["close"] +x*df[f"{signal}_{position}_{tp}_strength"] 
            df[f"{signal}_{position}_SL{tp[-1]}"] = df["close"] -x*penalty*abs(df[f"{signal}_{position}_{tp}_weakness"])

            # TIME TP AND SL
            df[f"{signal}_{position}_{tp}_t"] = df[f"{signal}_{position}_{tp}_strength_t"] 
            df[f"{signal}_{position}_SL{tp[-1]}_t"] = df[f"{signal}_{position}_{tp}_weakness_t"]

            # df[f"tide_{position}_{tp}"]=df[f"tide_{position}_{tp}"].fillna(method="ffill")

            # Risk reward ratio
            df[f"{signal}_{position}_RR{tp[-1]}"] = abs(df["close"]-df[f"{signal}_{position}_SL{tp[-1]}"]) / abs(df["close"] - df[f"{signal}_{position}_{tp}"])
            # df[f"{signal}_{position}_ub_RR{tp[-1]}"] = df[f"{signal}_{position}_RRRatio{tp[-1]}"].dropna().rolling(lookback).quantile(qtl)#-1
            # df[f"{signal}_{position}_lb_RR{tp[-1]}"] = df[f"{signal}_{position}_RRRatio{tp[-1]}"].dropna().rolling(lookback).quantile(1-qtl)#-1

    return df
# ============================================================================================================================================
# ============================================================================================================================================
#                               Tides Derivatives
# ============================================================================================================================================
# ============================================================================================================================================


def calc_tide_strengths(df0,
                        penalty = 1, # this widens the SL so that it is not hit too often
                        tp_position_dict = {"TP1": {"long":{"lookback":3, "qtl": 0.3}, 
                                                    "short": {"lookback":3, "qtl":0.3}
                                                    },
                                            "TP2": {"long":{"lookback":6, "qtl": 0.66}, 
                                                    "short": {"lookback":6, "qtl":0.66}
                                                    },
                                            "TP3": {"long":{"lookback":9, "qtl": 0.99}, 
                                                    "short": {"lookback":9, "qtl":0.99}
                                                    }
                                            }
                        ):
    df = df0.copy()


    # create a column to track the change in tide
    df['tide_change'] = df['tide'].diff().ne(0).cumsum()

    # create a column to count the duration of each tide
    df['tide_dur'] = df.groupby('tide_change').cumcount()+1
    # calculate percentile for tide_dur
    df['tide_dur'] = df["tide_dur"].shift(1)+1

    df.drop(columns=['tide_change'], inplace=True)

    df["tide_short_strength_t"] = np.where((df["tide"] > 0) & (df["S_rpnl"]>0), df['tide_dur'], np.nan)
    df["tide_short_weakness_t"] = np.where((df["tide"] > 0) & (df["S_rpnl"]<0), df['tide_dur'], np.nan)

    df["tide_long_strength_t"] = np.where((df["tide"] < 0) & (df["L_rpnl"]>0), df['tide_dur'], np.nan)
    df["tide_long_weakness_t"] = np.where((df["tide"] < 0) & (df["L_rpnl"]<0), df['tide_dur'], np.nan)

    # could have for multiple tide speeds, fast or slow
    df["tide_short_strength"] = np.where((df["tide"] > 0) & (df["S_rpnl"]>0), df["S_rpnl"], np.nan)
    df["tide_short_weakness"] = np.where((df["tide"] > 0) & (df["S_rpnl"]<0), df["S_rpnl"], np.nan)

    df["tide_long_strength"] = np.where((df["tide"] < 0) & (df["L_rpnl"]>0), df["L_rpnl"], np.nan)
    df["tide_long_weakness"] = np.where((df["tide"] < 0) & (df["L_rpnl"]<0), df["L_rpnl"], np.nan)
    # df["tide_short_str"] = np.where(df["tide"] > 0, df["S_pnl"], np.nan)
    # df["tide_long_str"] = np.where(df["tide"] < 0, df["L_pnl"], np.nan)

    # ALERT: tide_long_this should look at str_window amounts of tide-strengths, not str_window number of time periods
    # df["tide_short_z"] = calc_rolling_sr(df["tide_short_str"].dropna().values, window=str_window)
    # df["tide_long_z"] = calc_rolling_sr(df["tide_long_str"].dropna().values, window=str_window)

    for tp in ["TP1", "TP2", "TP3"]:
        for position in ["long", "short"]:
            # print(f"{tp} --> {position}")
            try:
                lookback = tp_position_dict[tp][position]["lookback"]
            except Exception as e:
                print(f"tp_position_dict[{tp}][{position}] does not exist: {e}")
                continue
            qtl = tp_position_dict[tp][position]["qtl"]
            df[f"tide_{position}_{tp}_strength"] = df[f"tide_{position}_strength"].dropna().rolling(lookback).quantile(qtl)
            df[f"tide_{position}_{tp}_strength"]=df[f"tide_{position}_{tp}_strength"].fillna(method="ffill")

            df[f"tide_{position}_{tp}_weakness"] = df[f"tide_{position}_weakness"].dropna().rolling(lookback).quantile(1-qtl)
            df[f"tide_{position}_{tp}_weakness"]=df[f"tide_{position}_{tp}_weakness"].fillna(method="ffill")

            df[f"tide_{position}_{tp}_strength_t"] = df[f"tide_{position}_strength_t"].dropna().rolling(lookback).quantile(qtl)#-1
            df[f"tide_{position}_{tp}_strength_t"]=df[f"tide_{position}_{tp}_strength_t"].fillna(method="ffill")

            df[f"tide_{position}_{tp}_weakness_t"] = df[f"tide_{position}_weakness_t"].dropna().rolling(lookback).quantile(1-qtl)#-1
            df[f"tide_{position}_{tp}_weakness_t"]= df[f"tide_{position}_{tp}_weakness_t"].fillna(method="ffill")

            if position == "long":
                x = 1 
            elif position == "short":
                x = -1

            # PRICE TP and SL
            df[f"tide_{position}_{tp}"] = df["close"] +x*df[f"tide_{position}_{tp}_strength"] 
            df[f"tide_{position}_SL{tp[-1]}"] = df["close"] - penalty*abs(df[f"tide_{position}_{tp}_weakness"])

            # TIME TP AND SL
            df[f"tide_{position}_{tp}_t"] = df[f"tide_{position}_{tp}_strength_t"] 
            df[f"tide_{position}_SL{tp[-1]}_t"] = df[f"tide_{position}_{tp}_weakness_t"]

            # df[f"tide_{position}_{tp}"]=df[f"tide_{position}_{tp}"].fillna(method="ffill")

            # Risk reward ratio
            df[f"tide_{position}_RRRatio{tp[-1]}"] = abs(df["close"]-df[f"tide_{position}_SL{tp[-1]}"])/abs(df["close"] - df[f"tide_{position}_{tp}"])
            df[f"tide_{position}_ub_RRRatio{tp[-1]}"] = df[f"tide_{position}_RRRatio{tp[-1]}"].dropna().rolling(lookback).quantile(qtl)#-1
            df[f"tide_{position}_lb_RRRatio{tp[-1]}"] = df[f"tide_{position}_RRRatio{tp[-1]}"].dropna().rolling(lookback).quantile(1-qtl)#-1

    return df


# ============================================================================================================================================
# ============================================================================================================================================
#                               Slopes
# ============================================================================================================================================
# ============================================================================================================================================

def calc_slopes(df0,
                slope_lengths:list=[7,10,14,20,28,40,56,80],
                scaling_factor:float = 1.0,
                lookback:int = 500,
                upper_quantile = 0.9,
                logRet_norm_window = 10,
                suffix=""):
    hour_List = np.array(slope_lengths) * scaling_factor
    min_List = [int(hour) for hour in hour_List]
    close = list(df0.filter(regex="close$").columns)[0]
    df = df0.copy()
    df_temp = pd.DataFrame()
    df_temp['logRet'] = np.log(1+df[close].pct_change())
    df_temp['logRet_norm'] = df_temp['logRet'] / df_temp['logRet'].rolling(logRet_norm_window).std()
    
    # we need to fillna(0) because some pockets have zero mobvement in price => o/0 = nan
    df_temp['logRet_norm'] = df_temp['logRet_norm'].fillna(0)
    df_temp['logLevel_norm'] = df_temp['logRet_norm'].rolling(lookback).sum()

    df_temp['logRet_norm'].isna().sum()
    df_temp['logLevel_norm'].isna().sum()

    slopeNames = []
    
    for minutes in min_List:
        slope_name = 'slope_' + str(minutes)
        slopeNames.append(slope_name)
        df_temp[slope_name] = (df_temp['logLevel_norm'] - df_temp['logLevel_norm'].shift(periods=minutes)) / minutes
        
    if suffix != "":
        suffix = "_"+suffix
    df[f'slope_avg{suffix}'] = df_temp[slopeNames].mean(axis=1, skipna=False)
    lower_quantile = round(1.0 - upper_quantile, 3)
    

    df[f'slope_u{suffix}'] = df[f'slope_avg{suffix}'].rolling(lookback).quantile(upper_quantile)
    df[f'slope_l{suffix}'] = df[f'slope_avg{suffix}'].rolling(lookback).quantile(lower_quantile)
    return df
    
    
#%% VOLUME PROFILE
def nb_vp():
    pass


#%% Tide metrics
def calc_tide_metrics(klines_indicators_dict):
    test = {}
    for instrument, df in klines_indicators_dict.items():
        df1 = df.copy()
        tide_labels = list(df1.filter(regex="tide$").columns) 
        mx_labels = list(df1.filter(regex="ebb$").columns)
        cols_to_get = ["1h_open","1h_high","1h_low","1h_close"] + tide_labels + mx_labels
        df1 = df1[cols_to_get].tail(2).copy()
        
        
        
        
        
        # RENAME AND TIDY UP
        for mx_label in mx_labels:
            tf,relabel = mx_label.split("_")
            df1.rename(columns={mx_label:f"{tf}_mx"},inplace=True)
        
        df1.reset_index(inplace=True)
        df1["instrument"] = instrument
        
        df1.set_index("instrument")
        df1 = df1.reset_index(drop=True)
        test[instrument] = df1
        

    final_df = pd.concat(test, axis=0)
    final_df.reset_index(drop=True, inplace=True)
    
    # final_df = final_df.set_index(keys="instrument")
    # final_df.sort_index(inplace=True)
    # multi_index = [(ins,dt) for ins,dt in final_df[["instrument", "date_time"]].to_dict().items()]
    # final_df.set_index(["instrument", "date_time"],inplace=True)
    final_df = final_df.round(2)
    tides = list(final_df.filter(regex="tide").columns)
    final_df[tides] = final_df.filter(regex="tide").astype(int)
    final_df.set_index(keys=["instrument", "date_time"],inplace=True)
    final_df.sort_index(inplace=True)
    
    s = final_df.style
    for idx, group_df in final_df.groupby('instrument'):
        s.set_table_styles({group_df.index[0]: [{'selector': '', 'props': 'border-top: 3px solid black;'}]}, 
                           overwrite=False, axis=1)

    mx_labels = list(final_df.filter(regex="mx$").columns)
    final_df1 = s.apply(tide_colors, axis=0, subset=list(final_df.filter(regex="tide").columns))
    # final_df1 = final_df.groupby("instrument").rank().style.background_gradient(subset=["1h_open","1h_high","1h_low","1h_close"]+mx_labels)
#%%
    return final_df1
    
#%%
def tide_colors(series):

    g = 'background-color: green;'
    r = 'background-color: orange;'
    w = ''

    return [r if e < 0 else g if e >0 else w for e in series]  


def ewmac(prices, window, alpha):
    ewma = np.empty(len(prices))
    ewma_old = prices[0]
    ewma[0] = ewma_old

    for i in range(1, len(prices)):
        ewma_old = ewma_old * (1 - alpha) + prices[i] * alpha
        ewma[i] = ewma_old

    return ewma


def calc_ewmac_sig(df0, 
             cols = ['open','high','low'],
             ewmac_window_func = lambda x: 288 * x,
             ewmac_alpha_func = lambda x: 2 * x,
             sig_name = "ewmac"):
    df = df0.copy() # if somehow need to save initial state of df before adding signals
    for col in cols:
        # print(f"col: {col}")
        np_col = df[col].values
        # print(f"len np_col: {len(np_col)}")
        ewmac_window = ewmac_window_func(i)
        ewmac_alpha = ewmac_alpha_func(i)
        ewmac_col = ewmac(np_col, ewmac_window, ewmac_alpha)
        df1 = df.copy()
        df1[f"{col}_{sig_name}"] = ewmac_col

        df = df1.copy()
    return df1



#%%

