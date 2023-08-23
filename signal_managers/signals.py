import numpy as np
import numba as nb
from tqdm import tqdm

@nb.njit(cache=True)
def rolling_zscore(np_col, 
                   window, 
                   threshold,
                   verbose=False):
    
    z = np.full(len(np_col), np.nan)
    sig = np.full(len(np_col), np.nan)
    for i in range(len(np_col)):
        if i >= window-1:
            np_col_i=np_col[i-window+1:i+1]
            np_col_i = np_col_i[~np.isnan(np_col_i)]
            ret = np.diff(np.log(np_col_i))
            # ret = ret[~np.isnan(ret)]

            if (len(ret) == 0) or (np.std(ret) == 0.0):
                res = 0
                z[i]=res
            else:
                res = (ret[-1] - np.mean(ret))/np.std(ret)
                z[i]=res

            res1 = np.sign(res) * np.floor(abs(res)) * (abs(res) >= threshold)
            sig[i] = res1
    return z,sig
    
def calc_z_sig(df0, 
             cols=['THB_USD_close','ES_USD_close'],
             window=288, # 24hours for 5min timeframe (should this lookback be reduced for downsampled timeframes)
             threshold=2,
             verbose=False,
             timeit=False,
             reciprocal=False,
             sig_name_1 = "z",
             sig_name_2 = "sig"):
    
    df=df0.copy()
    for col in cols:
        np_col = df[col].values
        if reciprocal:
            np_col = 1/np_col
        z,sig= rolling_zscore(np_col=np_col,
                              window=window,
                              threshold=threshold, 
                              verbose=verbose )
        df[sig_name_1] = z
        df[sig_name_2] = sig
        df[sig_name_2].replace(0, np.nan, inplace=True)
        df[sig_name_2].fillna(method="ffill",inplace=True)
        df[sig_name_2].fillna(0,inplace=True)
    return df
  

def populate_signals(instruments_dict,
                     cols=["close"],
                     base_freq = 5, # 5 minutes
                     base_window=288,
                     sig_name_1 = "z",
                     sig_name_2 = "sig"):
    
    for instrument,instrument_dict in tqdm(instruments_dict.items()):
        temp = {}
        for timeframe, df in instrument_dict.items():
            t = int(timeframe.split('Min')[0])
            
            if t == 5:
                window_t = base_window
            elif t>5:
                window_t = int(288 / (t/5))
            df = calc_z_sig(df,
                          cols=cols,
                          window=window_t, 
                          threshold=2,
                          verbose=False,
                          timeit=False,
                          sig_name_1=sig_name_1,
                          sig_name_2=sig_name_2)
            temp[timeframe] = df
            
        instruments_dict[instrument] = temp
    return instruments_dict

import numpy as np
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


from sklearn.decomposition import PCA
def calc_hedge_ratio(x):
    """
    x: pandas dataframe of 2 features
    """
    pca = PCA()
    pca_features = pca.fit_transform(x.dropna())
    beta = pca.components_[0 ,0] / pca.components_[1 ,0]
    return beta 


# =============================================================================
#  Merge instruments dict
# =============================================================================

import numpy as np
import pandas as pd

def merge_instruments_dict(instruments_dict,
                           spot_instrument: str,
                           futures_instruments: list,
                           futures_trading_times = ["02:30:00","10:00:00"],
                           trim_to_lagging_factor = True, #None
                           window = None, # ["2022-11-01": "2022-11-24 17:20:00"]
                           verbose = True):
    sig_dict = {}
    if verbose: print(f"spot: {spot_instrument}")
    if verbose: print(f"futures: {futures_instruments}")
    if verbose: print(f"trading time: {futures_trading_times}")
    
    signal_timeframes = list(instruments_dict[spot_instrument].keys())
    if verbose: print(f"{'='*40}\nMERGING\n{'='*40}")
    for timeframe in signal_timeframes:
        if verbose: print(f"\n{'-'*15}> {timeframe}")
        df = instruments_dict[spot_instrument][timeframe].copy()
        if verbose: print(f"df {timeframe} has duplicates? ---> {df.index.has_duplicates}")
        
        
        factors_label = []
        futures_label = []
        for instrument, instrument_dict in instruments_dict.items():
            if verbose: print(instrument)
            df_i = instrument_dict[timeframe].copy()
            if instrument == spot_instrument:
                if verbose: print(f"df_i {timeframe} has duplicates? ---> {df_i.index.has_duplicates}")
                continue

            else:
                if instrument in futures_instruments:
                    fut_label = instrument.split("_")[-1]
                    futures_label.append(fut_label)
                    if len(futures_trading_times)>0:
                        df_i = df_i.between_time(futures_trading_times[0],futures_trading_times[1])
                    df = pd.merge(df, df_i.add_suffix(f"_{fut_label}"), left_index=True, right_index=True, how="left")
                else:
                    factors_label.append(instrument)
                    df = pd.merge(df, df_i.add_suffix(f"_{instrument}"), left_index=True, right_index=True, how="left")
                if verbose: print(f"df after merging with {instrument} {timeframe} has duplicates? ---> {df.index.has_duplicates}") 
                    
        for futures_label_i in futures_label:
            # Should we only consider close?
            # How about midprice, or hc_spot - hc_fut? 
            df[f"spread_{futures_label_i}"] = np.where(df[f"volume_{futures_label_i}"]>0, df[f"close_{futures_label_i}"]-df["close"], np.nan)
            # df[f"spread_{futures_label_i}"] = np.where(df[f"volume_{futures_label_i}"]>0, df[[f"high_{futures_label_i}",f"low_{futures_label_i}"]].mean(axis=1)-df[["high","low"]].mean(axis=1), np.nan)
        
        df = df[~df.index.duplicated(keep='first')]
        if verbose: print(f"df {timeframe} has duplicates? ---> {df.index.has_duplicates}")
        
        if trim_to_lagging_factor:
            # TODO: HOW TO FIX THIS?! sometimes its not 2 nans for lagging factor, cld be only 1. but skip
            idx = df.fillna(method='ffill').dropna().index
            res_idx = df.loc[idx].fillna(method='bfill').dropna().index
            df_to_input = df.loc[res_idx].copy()
            
            # sig_dict[timeframe] = df.iloc[:-2,:].copy()
        else:
            df_to_input = df.copy()
            
        if window is None:
            sig_dict[timeframe] = df_to_input
        else:
            sig_dict[timeframe] = df_to_input[window[0]:window[1]]
        if verbose: print(f"{'='*40}\nEND MERGE\n{'='*40}\n\n")
        
    return sig_dict




def populate_z_signals(sig_dict, 
                       lookback_windows = [288,288], 
                       sig_threshold = 2,
                       columns = ["close_ES_USD", "spread_q"],
                       verbose=False,
                       take_reciprocal = False,
                       ):
    # =============================================================================
    # Signal Calculation
    # =============================================================================
    # if verbose: print(f"\n\n{'='*40}\nSIGNAL CALCULATION\n{'='*40}")
    output_dict = {}
    for timeframe, df in sig_dict.items():
        if verbose: print(f"\n{'-'*15}> {timeframe}")
        
        # =================================
        # Univariate z-score signal Calculation
        # =================================
        df1 = df.copy() 
        # if verbose: print("spread_q")
        # df1 = signals.calc_z_sig(df1,
        #                          cols=["spread_q"],
        #                          window=lookback_window, 
        #                          threshold = sig_threshold, 
        #                          reciprocal=take_reciprocal,
        #                          sig_name_1 = "z_q",
        #                          sig_name_2 = "sig_q")


        z_labels = []
        sig_labels = []
        for factor,lookback_window in zip(columns, lookback_windows):
            
            # lookback_window = lookback_window/int(timeframe[:-1])
            if verbose: print(f"Calculating z-score signal for {factor} --> {lookback_window}  ")
            df1 = calc_z_sig(df1,
                             cols=[factor],
                             window=lookback_window, 
                             threshold = sig_threshold,  
                             reciprocal = take_reciprocal,
                             sig_name_1 = f"z_{factor}",
                             sig_name_2 = f"sig_{factor}")
        output_dict[timeframe] = df1
    
    return output_dict

# =============================================================================
#  Merge and populate signals OLD
# =============================================================================
import pandas as pd
import numpy as np


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def populate_and_merge_signals(instruments_dict,
                               signal_timeframes = ["5Min","15Min","30Min", "60Min"],
                               spot_instrument: str = "USD_THB",
                               futures_instruments:list = ["USD_THB_f", "USD_THB_q"],
                               futures_trading_times:list = None,
                               factors = ["spread_q","z_ES"],
                               factors_to_normalize = ["spread_q","spread_f"],
                               to_probs = True,
                               target_signal = "pca_sig",
                               smallest_timeframe="5m",
                               lookback_window =288,
                               sig_threshold=2,
                               take_reciprocal=False,
                               L_uq = 0.95,
                               L_lq = 0.05,
                               L_q_lookback = 84, 
                               S_uq = 0.95,
                               S_lq = 0.05,
                               S_q_lookback = 84, 
                               pcs_dict = None,
                               train_window=["2020-01-01","2020-12-31"],
                               test_window=["2021-01-01","2022-12-31"],
                               verbose = False
                              ):
    
    
    
    # =============================================================================
    # Merge factors by timeframes
    # =============================================================================
    
    sig_dict = {}
    if verbose: print(f"{'='*40}\nMERGING\n{'='*40}")
    for timeframe in signal_timeframes:
        if verbose: print(f"\n{'-'*15}> {timeframe}")
        df = instruments_dict[spot_instrument][timeframe].copy()
        factors_label = []
        futures_label = []
        for instrument, instrument_dict in instruments_dict.items():
            if verbose: print(instrument)
            df_i = instrument_dict[timeframe].copy()
            if instrument == spot_instrument:
                continue

            else:
                if instrument in futures_instruments:
                    fut_label = instrument.split("_")[-1]
                    futures_label.append(fut_label)
                    if futures_trading_times is not None:
                        df_i = df_i.between_time(futures_trading_times[0],futures_trading_times[1])
                    df = pd.merge(df, df_i.add_suffix(f"_{fut_label}"), left_index=True, right_index=True, how="left")
                else:
                    factor_label = instrument.split("_")[0]
                    factors_label.append(factor_label)
                    df = pd.merge(df, df_i.add_suffix(f"_{factor_label}"), left_index=True, right_index=True, how="left")

        for futures_label_i in futures_label:
            df[f"spread_{futures_label_i}"] = np.where(df[f"volume_{futures_label_i}"]>0, df[f"close_{futures_label_i}"]-df["close"], np.nan)

        sig_dict[timeframe] = df
    
        if verbose: print(f"futures labels: {futures_label}")
        if verbose: print(f"factors labels: {factors_label}")
        
    # =============================================================================
    # Signal Calculation
    # =============================================================================
    if verbose: print(f"\n\n{'='*40}\nSIGNAL CALCULATION\n{'='*40}")
    for timeframe, df in sig_dict.items():
        if verbose: print(f"\n{'-'*15}> {timeframe}")
        
        # =================================
        # Univariate z-score signal Calculation
        # =================================
        df1 = df.copy()  
        # if verbose: print("spread_q")
        # df1 = signals.calc_z_sig(df1,
        #                          cols=["spread_q"],
        #                          window=lookback_window, 
        #                          threshold = sig_threshold, 
        #                          reciprocal=take_reciprocal,
        #                          sig_name_1 = "z_q",
        #                          sig_name_2 = "sig_q")


        z_labels = []
        sig_labels = []
        for factor in factors_label:
            if verbose: print(f"Calculating z-score signal for {factor} ... ")
            df1 = calc_z_sig(df1,
                             cols=[f"close_{factor}"],
                             window=lookback_window, 
                             threshold = sig_threshold,  
                             reciprocal=take_reciprocal,
                             sig_name_1 = f"z_{factor}",
                             sig_name_2 = f"sig_{factor}")

            z_labels.append(f"z_{factor}")
            sig_labels.append(f"sig_{factor}")


        # =================================
        # Composite signal Calculation
        # =================================
    
        if len(factors) == 1:
            df_total = df1.copy()
            
        elif (pcs_dict is None) and len(factors) >1:
            if verbose: print(f"Calculating PC weights for \nfactors: {factors}")

            # train_window=["2020-01-01","2020-12-31"],
            # test_window=["2021-01-01","2022-12-31"],
                               
            df_train = df1[train_window[0]:train_window[1]].copy()
            df_test = df1[test_window[0]:test_window[1]].copy()

            # Normalise spreads 
            if factors_to_normalize is not None:
                factors_z = []
                for factor_to_normalize in factors_to_normalize:
                    scaler = StandardScaler()
                    df_train[f"{factor_to_normalize}_norm"] = scaler.fit_transform(df_train[[factor_to_normalize]])
                    df_test[f"{factor_to_normalize}_norm"] = scaler.transform(df_test[[factor_to_normalize]])
                    factors_z.append(f"{factor_to_normalize}_norm")
                    
                non_normalized_factors = [i for i in factors if i not in factors_to_normalize]
                factors_to_comp = factors_z + non_normalized_factors
                if verbose: print(f"---> {factors_to_comp}") 
            else:
                factors_to_comp = factors
                
            pca=PCA()
            df_sig_pca = pca.fit_transform(df_train[factors_to_comp].dropna())
            # pcs = pca.explained_variance_ratio_
            pcs = pca.components_[:,0]
            if verbose: print(f"pcs: --> {pcs} ----> {pca.components_[:,0]}")
            df_train["comp_sig"] = (df_train.loc[:,factors_to_comp]*pcs).sum(axis=1)
            df_test["comp_sig"] = (df_test.loc[:,factors_to_comp]*pcs).sum(axis=1)
            df_total = pd.concat([df_train, df_test],axis=0)
            
            
            
        
        else: 
            df1["comp_sig"] = (df1.loc[:,factors_to_comp]*pcs_dict[timeframe]).mean(axis=1)
            
            df_total = df1.copy()
            
            
            
        # =================================
        # Signal threshold Calculation
        # =================================
        if to_probs == "sigmoid":
            df_total[target_signal] = calc_sigmoid(df_total[target_signal].values)
        elif to_probs == "tanh":
            df_total[target_signal] = calc_tanh(df_total[target_signal].values)
            
        df_total[f'L_uq'] = df_total[target_signal].rolling(L_q_lookback).quantile(L_uq).shift(1)
        df_total[f'L_lq'] = df_total[target_signal].rolling(L_q_lookback).quantile(L_lq).shift(1)
        
        df_total[f'S_uq'] = df_total[target_signal].rolling(S_q_lookback).quantile(S_uq).shift(1)
        df_total[f'S_lq'] = df_total[target_signal].rolling(S_q_lookback).quantile(S_lq).shift(1)
        
        sig_dict[timeframe] = df_total

    instruments_dict["sig"] = sig_dict
        
    # =============================================================================
    # MERGE ALL TIMEFRAMES
    # =============================================================================  
    df= sig_dict[smallest_timeframe]
    signals_to_query = [target_signal, "L_uq", "L_lq", "S_uq", "S_lq"]
    
    df = df.add_prefix(f"{smallest_timeframe}_")
    for timeframe, df_i in sig_dict.items():
        if timeframe == smallest_timeframe:
            continue
        if verbose: print(f"merging: {timeframe}")
        df = pd.merge(df, df_i[signals_to_query].add_prefix(f"{timeframe}_"), how="left", left_index=True, right_index=True)
            
        
        
    return df, sig_dict
  

    # Should i merge all? or just merge the signals
    # on that note, should calculate quantile thresholds here! 
    
    
    
    
    
