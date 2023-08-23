from os import listdir
from utils import pickle_helper
from signals import resampler


def load_data(instruments = "ALL",#["BTC_USD","BTCPERP_USD","ETH_USD","ETHPERP_USD","SPY_USD"],
              instrument_to_trade="BTC_USD",
              base_freq = "5m",
              resample_to = ["30m","60m"],
              path:str="./database/instruments/",
              fix_tz=None, #{"from":"US/Central", "to":"UTC", "pickle":True}:
              set_quote_USD = False,    
              ):
    """
    For each instrument, resample to desired timeframes 
    
    then store in dictionaries?? or merge them cross sectionally??
    
    """
    if len(resample_to) > 0:
        for t in resample_to:
            assert "m" in t
        assert "m" in base_freq
    
    # =============================================================================
    # pickle load instruments
    # =============================================================================
    pickled_instruments_list = listdir(path)
    pickled_instruments_list = [i.split(".")[0] for i in pickled_instruments_list]
    instruments_dict = {}
    
    if instruments == "ALL":
        instruments_to_get = pickled_instruments_list
    else:
        instruments_to_get = instruments
    for instrument in instruments_to_get:
        
        if instrument in pickled_instruments_list:
            df = pickle_helper.pickle_this(data=None, pickle_name=instrument, path=path)
            
            if set_quote_USD and instrument.split("_")[1] != "USD":
                instrument = f"{instrument.split('_')[1]}_USD"
                df[["open","high","low","close"]]=1/df[["open","high","low","close"]]
                pickle_helper.pickle_this(df, pickle_name=f"{instrument}", path=path)  
                
                
                
            if fix_tz is not None:
                df.index = df.index.tz_localize(fix_tz["from"])#.tz_convert('UTC')
                df.index = df.index.tz_convert(fix_tz["to"])
                df.index = df.index.tz_convert(None)
                df["close_time"]=df.index.astype(int)// (10 ** 9)
                if fix_tz["pickle"]:
                    pickle_helper.pickle_this(data=df, pickle_name=instrument, path=path)
                
            print(f"{instrument} \n   range: {df.index[0]} to {df.index[-1]} \n   rows: {len(df)}")
            # Ensure in standardised format (open, high low, close, volume, close_time)
            if "closeTime" in df.columns:
                print(f"{instrument}--> found closeTime")
                df.rename(columns={"closeTime":"close_time"},inplace=True)
                pickle_helper.pickle_this(data=df, pickle_name=instrument, path=path)
                
            # RESAMPLE 
            resampled_dict = {}
            resampled_dict[base_freq] = df
            # print(f"   --> {base_freq}")
            if len(resample_to) >0:
                for t in resample_to:
                    # print(f"   --> {t}")
                    freq_int = int(t.split("m")[0]) / int(base_freq.split("m")[0])
                    freq_int = int(freq_int)
                    try:
                        resampled_dict[t] = resampler.calc_klines_resample(df,window=freq_int, resample_to=t)
                    except Exception as e:
                        return df
            instruments_dict[instrument] = resampled_dict
            
    # =============================================================================
    # Merge them with instrument_to_trade on left 
    # =============================================================================
    # if instrument_to_trade is not None:
        
    return instruments_dict


    
    