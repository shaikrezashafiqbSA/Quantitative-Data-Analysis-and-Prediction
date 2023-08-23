import numpy as np
import pandas as pd
import numba as nb
# from strategy.resampler import calc_ohlc_from_series
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

# ======================================================================
#                               MFI
# ======================================================================

def rolling_mfi(np_col: np.array,
                 MFI_window_threshold_func,
                 fixed_window:bool=False,
                 tolerance = 1e-7,
                 fill_value = np.nan) -> np.array:
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
    n = len(np_col)
    MFI_list = []
    for i in range(n):
        windows = MFI_window_threshold_func(np_col, i)
        MFI_i_np = np.full((len(windows)), np.nan)
        psum = np.full((len(windows)), np.nan)
        nsum = np.full((len(windows)), np.nan)
        for w, window in enumerate(windows):
            max_lookback = window # this could be list
            if fixed_window:
                high_i = np_col[max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
                low_i = np_col[max_lookback:i+1,3]
                close_i = np_col[max_lookback:i+1,4]
                volume_i = np_col[max_lookback:i+1,-1]
            else:
                # expanding window
                high_i = np_col[:,2]
                low_i = np_col[:,3]
                close_i = np_col[:,4]
                volume_i = np_col[:,-1]
            typicalPrice = (high_i+low_i+close_i) / 3
            rawMoneyFlow = typicalPrice * volume_i

            psum = np.full(n, np.nan)
            nsum = np.full(n, np.nan)

            psum[w] = 0
            nsum[w] = 0
            for j in range(int(window)):
                psum[w] += rawMoneyFlow[i-j]
                nsum[w] += rawMoneyFlow[i-j]
            # mfi = 100 * psum[w]/(psum[w]+nsum[w]) # this one yield invalid value error 

            # Create masks for NaN, Inf, and -Inf values
            nan_mask = np.isnan(psum) | np.isnan(nsum)
            inf_mask = np.isinf(psum) | np.isinf(nsum)

            # Replace NaN, Inf, and -Inf values with np.nan
            psum_clean = np.where(nan_mask | inf_mask, np.nan, psum)
            nsum_clean = np.where(nan_mask | inf_mask, np.nan, nsum)


            denominator = psum_clean[w] + nsum_clean[w]
            numerator = 100 * psum_clean[w]
            mfi = np.where(np.abs(denominator) > tolerance, numerator / denominator, np.where(np.abs(denominator) < tolerance, np.inf, -np.inf))


            # solution in code: https://stackoverflow.com/questions/31323499/numpy-where-function-returns-invalid-value-encountered-in-divide
            

            MFI_i_np[w] = mfi
        MFI_list.append(MFI_i_np)

    return MFI_list


def MFI_window_threshold_func(x, i):
    """
    Define a more complex function to calculate the window lengths, thresholds, and sensitivity parameters
    based on characteristics of the input data.

    Args:
        x: x is a np_array of cols + dynamic_param_col
        i: The index of the current data point.

    Returns:
        The list of window lengths to use.
        The list of thresholds to use for each window length.
        The sensitivity parameter.
    """

    # Calculate the standard deviation of the price data.
    std = np.std(x)

    # Calculate the minimum window length.
    min_window_length = int(std * 10)

    # Calculate the maximum window length.
    max_window_length = int(std * 100)

    # Calculate the number of thresholds to use.

    # Generate a list of possible window lengths.
    window_lengths = np.linspace(min_window_length, max_window_length, num=10)

    return window_lengths
    

def calc_mfi_sig(df0, 
                 cols_set = [['high','low', 'close', 'volume'], ['high','low', 'open', 'volume']],
                 MFI_window_threshold_func = MFI_window_threshold_func,
                 dynamic_param_col = None,
                 fixed_window = False):  # fixed_window can also be dynamicised away by VARYING expanding window based on volatility
   
    df = df0.copy() # if somehow need to save initial state of df before adding signals
    df["date_time"] = df.index
    for cols in tqdm(cols_set):
        # print(f"col: {col}")
        if dynamic_param_col is not None:
            np_col = df[['date_time']+cols+dynamic_param_col].values
        else:
            np_col = df[['date_time']+cols].values
    
        # print(f"len np_col: {len(np_col)}")
        mfi = rolling_mfi(np_col,
                          MFI_window_threshold_func = MFI_window_threshold_func,
                          fixed_window = fixed_window,)
        # print(f"shape tide: {np.shape(tide)}, shape ebb: {np.shape(ebb)}, shape flow: {np.shape(flow)}")
        df1 = df.copy()
        # print(f"len df1: {len(df1)}")
        # if window_threshold_func() 
        
        for t,i in zip(df1.index, range(len(df1))):
            # print(f"t: {t}, i: {i}")
            windows = tide_window_threshold_func(np_col, i)
            # print(f"i-{i}: post processing --> tide shape: {np.shape(tide[i])} --> {tide[i]}")
            for w, window in enumerate(windows):
                df1.at[t, f"MFI{window}"] = mfi[i][w]

        df = df1.copy()
    return df



# ======================================================================
#                               Tides
# ======================================================================
    
# @nb.njit(cache=True)
def rolling_sum(heights, w=4):
    ret = np.cumsum(heights)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:]


# @nb.njit(cache=True)
def calc_exponential_height(heights, w):  ## CHECK!!
    # heights = OHLCVT_array[:,1]-OHLCVT_array[:,2]
    rolling_sum_H_L = rolling_sum(heights, w)
    # rolling_sum_H_L = np.full(len(heights),np.nan)
    # for idx in range(window-1,len(heights)):
    #     # print(f"summing {start_idx} to {idx}")
    #     rolling_sum_H_L[idx] = np.sum(heights[idx-window+1:idx])
    # mpbw=(heights.rolling(window=w).sum())
    exp_height = (rolling_sum_H_L[-1] - heights[-w] + heights[-1]) / w
    return exp_height  # (mpbw.iloc[-1]-heights[-w]+heights[-1])/w


# @nb.njit(cache=True)
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

    # undertow = [[1 if new_open > previous_ebb else 0] if previous_tide == 0 else [1 if new_open < previous_ebb else 0]][0][0]
    " undertow "
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

    # surftow = [[1 if new_high > previous_ebb else 0] if previous_tide == 0 else [1 if new_low < previous_ebb else 0]][0][0]
    " surftow "
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

    " Calculate change in tide: flow"

    # heights = df["high_i"][-67:]-df["low_i"][-67:]
    heights = high_i - low_i
    heights = heights[-max(windows):]

    w_0 = 0
    for w in windows:
        w_i = calc_exponential_height(heights, w)
        if w_i > w_0:
            max_exp_height = w_i
            w_0 = w_i
    # max_exp_height=max([calc_exponential_height(heights,w) for w in windows])#calc_exponential_height(prices,lengths[0]),calc_exponential_height(prices,lengths[1]),calc_exponential_height(prices,lengths[2]))    #THIS CAN BE CHANGED TO separate rolling functions#

    # sensitivity=sensitivity/100
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

    " flow "
    # flow = [previous_ebb+additive if previous_tide == 1 else previous_ebb-additive][0]
    if previous_tide:
        flow = previous_ebb + additive
    else:
        flow = previous_ebb - additive

    " interim tides "
    # tide_1 = [1 if new_open >= flow else 0][0]
    if new_open >= flow:
        tide_1 = 1
    else:
        tide_1 = 0

    # tide_2 = [[1 if new_low < previous_ebb else 0] if tide_1 == 1 else [1 if new_high > previous_ebb else 0]][0][0]
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

    # tide_3 =[[1 if surftow == 1 else 0] if tide_2 ==1 else [[ 0 if undertow == 1 else 1] if surftow == 0 else 0]][0][0]
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

    # tide_4 = [[1 if new_low>=flow else 0] if tide_1 == 1 else [1 if new_high > flow else 0]][0][0]
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

    " tide formulation "
    if tide_1 == 1 and tide_4 == 1:
        new_tide = 1
    elif tide_1 == 0 and tide_4 == 0:
        new_tide = 0
    else:
        new_tide = previous_tide

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
def rolling_tide(np_col,
            fixed_window: bool,
            tide_window_threshold_func):

    n = len(np_col)
    
    previous_tide = np.nan
    previous_ebb = np.nan
    previous_flow = np.nan
    
    windows_sets, thresholds, sensitivities=tide_window_threshold_func(np_col,0)
    tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
    ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
    flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)

    max_lookback = np.max(windows_sets)
    tide_list = [tide_i_np]*max_lookback
    ebb_list = [ebb_i_np]*max_lookback
    flow_list = [flow_i_np]*max_lookback
    for i in range(max_lookback, n + 1):
        windows_sets, thresholds, sensitivities = tide_window_threshold_func(np_col,i)
        tide_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        ebb_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        flow_i_np = np.full((len(windows_sets), len(thresholds)), np.nan)
        # print(f"shape tide: {np.shape(tide_i_np)}, shape ebb: {np.shape(ebb_i_np)}, shape flow: {np.shape(flow_i_np)}")
        for w,windows in enumerate(windows_sets):
            for th,threshold in enumerate(thresholds):
                if fixed_window:
                    open_i = np_col[-max_lookback:i+1,1]
                    high_i = np_col[-max_lookback:i+1,2] # y2 = x[-max_lookback:,2]
                    low_i = np_col[-max_lookback:i+1,3]
                else:
                    # expanding window
                    open_i = np_col[:,1]
                    high_i = np_col[:,2] 
                    low_i = np_col[:,3]

                try:
                    previous_tide=tide_list[i-1][th,w]
                    previous_ebb=ebb_list[i-1][th,w]
                    previous_flow=flow_list[i-1][th,w]
                except Exception as e:
                    previous_tide = np.nan
                    previous_ebb = np.nan
                    previous_flow = np.nan
                    # print(f"t={t}, all still nan...")
                windows = np.array(windows)
                threshold = int(threshold)
                sensitivity = float(sensitivities[0])
                tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
                                                high_i=high_i,
                                                low_i=low_i,
                                                previous_tide=previous_tide,
                                                previous_ebb=previous_ebb,
                                                previous_flow=previous_flow,
                                                windows=windows,
                                                thresholds=threshold,
                                                sensitivity=sensitivity) # to be generalised next
                # print(f"t:{t} w:{w} th:{th} ---> windows={windows}, threshold={threshold}, sensitivity={sensitivities[0]}")
                
                tide_i_np[th, w] = tide_i
                ebb_i_np[th, w] = ebb_i
                flow_i_np[th, w] = flow_i

                # print(f"shape tide: {np.shape(tide_i_np)}, shape ebb: {np.shape(ebb_i_np)}, shape flow: {np.shape(flow_i_np)}")
        tide_list.append(tide_i_np)
        ebb_list.append(ebb_i_np)
        flow_list.append(flow_i_np)
        
    return tide_list,ebb_list,flow_list


# ======================================================================
#                               tide V2
# ======================================================================
def tide_window_threshold_func_adv(x, i):
    """
    Define a more complex function to calculate the window lengths, thresholds, and sensitivity parameters
    based on characteristics of the input data.

    Args:
        x: x is a np_array of cols + dynamic_param_col
        i: The index of the current data point.

    Returns:
        The list of window lengths to use.
        The list of thresholds to use for each window length.
        The sensitivity parameter.
    """

    # Calculate the standard deviation of the price data.
    std = np.std(x)

    # Calculate the minimum window length.
    min_window_length = int(std * 10)

    # Calculate the maximum window length.
    max_window_length = int(std * 100)

    # Calculate the number of thresholds to use.
    num_thresholds = 3

    # Generate a list of possible window lengths.
    window_lengths = np.linspace(min_window_length, max_window_length, num=10)

    # Generate a list of possible thresholds.
    thresholds = np.linspace(0, 1, num=num_thresholds)

    # Calculate the sensitivity parameter.
    sensitivity = 2

    return window_lengths, thresholds, sensitivity


def tide_window_threshold_func(x, i):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    window_lengths = [[5, 20, 67]]
    thresholds = [0.1]
    sensitivity = [50]
    return window_lengths, thresholds, sensitivity



def calc_tide_sig(df0, 
             cols_set = [['open','high','low']],
             tide_window_threshold_func = tide_window_threshold_func,
             sig_name_1 = "tide",
             sig_name_2 = "ebb",
             sig_name_3 = "flow",
             fixed_window=False,
            dynamic_param_col = None ):
    """
    Calculate the tide, ebb, and flow signals for a data set.

    Args:
        df0: The data set.
        cols_set: The list of columns to use for the signal.
        tide_window_threshold_func: The function to calculate the window lengths, thresholds, and sensitivity parameters.
        sig_name_1: The name of the column for the tide signal.
        sig_name_2: The name of the column for the ebb signal.
        sig_name_3: The name of the column for the flow signal.

    Returns:
        A new data set with the tide, ebb, and flow signals added.
    """

    df = df0.copy() # if somehow need to save initial state of df before adding signals
    df["date_time"] = df.index
    for cols in tqdm(cols_set):
        # print(f"col: {col}")
        if dynamic_param_col is not None:
            np_col = df[['date_time']+cols+dynamic_param_col].values
        else:
            np_col = df[['date_time']+cols].values
        np_col = np_col[:i] # THIS COULD BE SOURCE OF POTENTIAL POINTER / COPY ERROR 

        tide,ebb,flow = rolling_tide(np_col,
                                     tide_window_threshold_func = tide_window_threshold_func,
                                     fixed_window = fixed_window,)
        # print(f"shape tide: {np.shape(tide)}, shape ebb: {np.shape(ebb)}, shape flow: {np.shape(flow)}")
        df1 = df.copy()
        # print(f"len df1: {len(df1)}")
        # if window_threshold_func() 
        for t,i in zip(df1.index, range(len(df1))):
            # print(f"t: {t}, i: {i}")
            windows, thresholds, sensitivities = tide_window_threshold_func(np_col, i)
            # print(f"i-{i}: post processing --> tide shape: {np.shape(tide[i])} --> {tide[i]}")
            for w, window in enumerate(windows):
                for th, threshold in enumerate(thresholds):
                    window_label = "-".join([f"{i}" for i in window])
                    df1.at[t, f"{sig_name_1}_w{window_label}t{threshold}"] = tide[i][th, w]
                    df1.at[t, f"{sig_name_2}_w{window_label}t{threshold}"] = ebb[i][th, w]
                    df1.at[t, f"{sig_name_3}_w{window_label}t{threshold}"] = flow[i][th, w]

        df = df1.copy()
    return df1
    

# ======================================================================
#                               z signal
# ======================================================================
def window_threshold_func_vol(x, i):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    # Calculate the volatility of the input data
    volatility = np.std(x) # This is too smooth to be vol triggers have to change for future use
    
    # Set the window lengths and threshold values based on the volatility
    if volatility < 0.01:
        windows = [288, 576]
        thresholds = [1.5, 2, 2.5]
    elif volatility < 0.05:
        windows = [144, 288]
        thresholds = [1, 1.5, 2]
    else:
        windows = [72, 144]
        thresholds = [0.5, 1, 1.5]
    
    return windows, thresholds

# @nb.njit(cache=True)
def window_threshold_func(x, i):
    """
    Template for dynamic parameters function
    where x is a np_array of cols + dynamic_param_col
    """
    return [288], [2]
    # return [3,5,8,13,21,81,288], [2,1,3,4,5,6,7,8]

# @nb.njit(cache=True)
def rolling_zscore(np_col, window_threshold_func):
    z = np.full(len(np_col), np.nan)
    sigs = []
    # print(f"len(np_col): {len(np_col)}")
    for i in range(len(np_col)):
        windows, thresholds = window_threshold_func(np_col, i)
        sigs_i = np.full((len(windows), len(thresholds)), np.nan)
        for k, window in enumerate(windows):
            if i >= window-1:
                np_col_i=np_col[i-window+1:i+1]
                np_col_i = np_col_i[~np.isnan(np_col_i)]
                ret = np.diff(np.log(np_col_i))

                if (len(ret) == 0) or (np.std(ret) == 0.0):
                    res = 0
                    z[i]=res
                else:
                    res = (ret[-1] - np.mean(ret))/np.std(ret)
                    z[i]=res

                for j, threshold in enumerate(thresholds):
                    res1 = np.sign(res) * np.floor(abs(res)) * (abs(res) >= threshold)
                    sigs_i[k, j] = res1
        sigs.append(sigs_i)
    return z,sigs

from tqdm import tqdm

def calc_z_sig(df0, 
             cols = ['open','high','low'],
             window_threshold_func = lambda x, i: ([3,5,8,13,21,81,288], [2,1,3,4,5,6,7,8]), # can this work?! wouldnt python see , i as another new param to function? 
             sig_name_1 = "z",
             sig_name_2 = "sig",
             dynamic_param_col=None):
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
    for col in tqdm(cols):
        # print(f"col: {col}")
        if dynamic_param_col is not None:
            np_col = df[cols+dynamic_param_col].values
        else:
            np_col = df[cols]
        # print(f"len np_col: {len(np_col)}")
        z,sigs= rolling_zscore(np_col=np_col,
                              window_threshold_func = window_threshold_func
                              )

        df1 = df.copy()
        df1[f"{col}_{sig_name_1}"] = z
        # print(f"shape z: {np.shape(z)}, shape sigs: {np.shape(sigs)}, len df: {len(df1)}")
        # if window_threshold_func() 
        for t,i in zip(df1.index, range(len(df1))):
            windows, thresholds = window_threshold_func(df1.iloc[i], i)
            for k, window in enumerate(windows):
                for j, threshold in enumerate(thresholds):
                    df1.at[t, f"{col}_{sig_name_2}_w{window}t{threshold}"] = sigs[i][k, j]

        df = df1.copy()
    return df1



# 
# ======================================================================
#                               Tide derivatives
# ======================================================================

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


# ======================================================================
#                               SLOPES
# ======================================================================
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
    for col in tqdm(cols):
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

