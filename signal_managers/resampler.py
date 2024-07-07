import numpy as np
import pandas as pd
import numba as nb


@nb.njit(cache=True)
def continuous_resampling(closetime, open_, high, low, close, vol, x):
    x=x-1
    n = len(close)
    r_closetime = np.full(n,np.nan)
    r_open = np.full(n, np.nan)
    r_high = np.full(n, np.nan)
    r_low = np.full(n, np.nan)
    r_close = np.full(n, np.nan)
    r_vol = np.full(n, np.nan)

    for i in range(x,n):
        r_closetime[i] = closetime[i]
        r_open[i]=open_[i-x]
        r_high[i]=np.max(high[i-x:i+1])
        r_low[i] = np.min(low[i-x:i+1])
        r_close[i] = close[i]
        r_vol[i] = np.sum(vol[i-x:i+1])
    
    return r_closetime,r_open,r_high,r_low,r_close,r_vol

def calc_klines_resample(df,
                         window, 
                         instrument_name = None, # BTC_USD
                         resample_to=None # "12T" for 5min base and want to resample to 1hr
                         ):
    cols = list(df.columns)
    
    if instrument_name is not None:
        instrument_name = instrument_name + "_"
    else:
        instrument_name = ""
        
        
    for required_col in ["open","high","low","close","volume","close_time"]:
        assert f"{instrument_name}{required_col}" in cols
        
    closeTime = df[f"{instrument_name}close_time"].to_numpy()
    open_ = df[f"{instrument_name}open"].to_numpy()
    high = df[f"{instrument_name}high"].to_numpy()
    low = df[f"{instrument_name}low"].to_numpy()
    close = df[f"{instrument_name}close"].to_numpy()
    vol = df[f"{instrument_name}volume"].to_numpy()
    
    r_closetime,r_open,r_high,r_low,r_close,r_vol = continuous_resampling(closeTime,open_, high, low, close, vol, window)
    
    
    test= pd.DataFrame({f"{instrument_name}open":r_open,
                        f"{instrument_name}high":r_high,
                        f"{instrument_name}low":r_low,
                        f"{instrument_name}close":r_close,
                        f"{instrument_name}volume":r_vol,
                        f"{instrument_name}close_time": r_closetime})
    # test['date_time'] = pd.to_datetime(test['closeTime'], unit='s').round("1T")
    test.index = df.index
    
    if resample_to is not None :
            test = test.resample(f"{resample_to}", label='right', closed='right').last()

    return test

    
def resample_instruments_dict(instruments_dict,
                              resample_to_list = ["2h", "3h", "4h", "6h", "12h"],
                              first_timeframe = "1h",
                              clean_base_data_flag = False):
    # Resample from 1hr --> 2h, 3h, 4h, 6h, 12h, 24h, 

    # resample_to_list = ["15m", "20m", "30m", "60m"]
    # first_timeframe = "5m"
    for instrument,instrument_dict in instruments_dict.items():
        # print(f"\n{instrument}\n")
        cleaned_flag = False
        for resample_to in resample_to_list:
            # only clean_base_data for the first timeframe
            if clean_base_data_flag and not cleaned_flag:
                instrument_dict[first_timeframe] = clean_base_data(instrument_dict[first_timeframe])
                cleaned_flag = True
            df = instrument_dict[first_timeframe].copy()
            window = int(int(resample_to[:-1])/int(first_timeframe[:-1]))
            # print(f"Resampling {first_timeframe} to {resample_to} --> window: {window}")
            if "m" in resample_to:
                resample_to_formatted = resample_to[:-1]+"T"
            else:
                resample_to_formatted = resample_to
            df_resampled = calc_klines_resample(df,window=window, resample_to=resample_to_formatted)
            df_resampled.fillna(method="ffill", inplace=True)
            df_resampled.dropna(inplace=True)
            # print(df_resampled.columns)
            instruments_dict[instrument][resample_to] = df_resampled
    return instruments_dict
    

def clean_base_data(df0, timeframe="1T"):

    df = df0.resample(timeframe).last()

    # Forward fill 'close' column
    df['close'] = df['close'].ffill()

    # Fill 'open', 'high', 'low' with 'close'
    df[['open', 'high', 'low']] = df[['open', 'high', 'low']].apply(lambda row: row.fillna(df['close']))

    # Fill 'volume' with 0
    df['volume'] = df['volume'].fillna(0)

    # Fill 'close_time' with timestamp from 'datetime'
    df['close_time'] = df.index.values.astype(np.int64) // 10 ** 6

    return df