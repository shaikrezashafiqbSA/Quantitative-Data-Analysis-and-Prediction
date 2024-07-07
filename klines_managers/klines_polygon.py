import pandas as pd
import numpy as np
import time
from datetime import datetime

from polygon import RESTClient
from utils import pickle_helper
from settings import POLYGON_API_KEY

timespan_map = {"m":"minute",
                "h":"hour",
                "d":"day",
                "w":"week",
                "M":"month"}
timeframe_to_seconds = {"m":60,
                        "h":3600,
                        "d":86400,
                        "w":604800,
                        "M":2592000}

import datetime
def string_to_milliseconds(date_string):
    date_time = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return int(date_time.timestamp() * 1000)

def milliseconds_to_datestring(milliseconds):
    seconds = milliseconds / 1000
    date_time = datetime.datetime.fromtimestamp(seconds)
    return date_time.strftime('%Y-%m-%d %H:%M:%S')

class KlinesManagerPolygon:
    def __init__(self, 
                 api_key=POLYGON_API_KEY,
                 database_path = "./database/klines/polygon/"):
        self.client = RESTClient(api_key=api_key)
        self.database_path = database_path

    def get_ohlcv(self,
                  symbol, 
                  timeframe,
                  since,
                  limit=5000,
                  max_retries = 3,
                  verbose = True,
                  to="2100-12-31"):
        num_retries = 0
        sleep_length = 10000
        multiplier=int(timeframe[:-1])
        timespan=timespan_map[timeframe[-1]]
        while num_retries < max_retries:
            try: 
                # if verbose: print(f"Loading {symbol} {multiplier} {timespan} from {since}")
                klines = self.client.get_aggs(ticker=symbol,
                                              multiplier=multiplier, #1h --> h
                                              timespan=timespan,     #1h --> 1
                                              from_= since, 
                                              to=to) 
            except Exception as e:
                num_retries += 1
                print(f"{e} \n---> retry attempt: {num_retries}")
                time.sleep(sleep_length)
                sleep_length+=10000
            else:
                return klines
            
    def get_ohlcvs_paginated(self,
                             symbol = "US500",
                             timeframe = "1h",
                             since = "2022-12-01 00:00:00",
                             limit = 1000,
                             max_retries=300,
                             verbose = True,
                             rate_limit_buffer_ms = 1):
        
        
        # convert since from string to milliseconds integer if needed
        if isinstance(since, str):
            since = string_to_milliseconds(since)

        # timestamp now
        until_timestamp = int(time.time() * 1000)

        # timeframe (eg: 1h) to timestamp delta
        timeframe_duration_in_seconds = int(timeframe[:-1]) * timeframe_to_seconds[timeframe[-1]]
        timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
        
        # timedelta to paginate by 
        # limit to low limit rest call if since and 
        timedelta = limit * timeframe_duration_in_ms
        print(f"until_timestamp: {milliseconds_to_datestring(until_timestamp)}")
        # print(f"timeframe_duration_in_ms: {timeframe_duration_in_ms}")
        # print(f"timeframe_duration_in_ms: {timeframe_duration_in_ms}")
        # print(f"timedelta: {timedelta}")
        all_ohlcv = []
        fetch_since = since
        while True:
            # Fetch backwards from since date all the way till current (opposite of ccxt paginator)
            
            # if verbose: print(f"fetching {timeframe} bars from {milliseconds_to_datestring(fetch_since)}")
            ohlcv = self.get_ohlcv(symbol, timeframe, fetch_since, limit, max_retries)
            if len(ohlcv) == 0:
                break
            fetched_since = ohlcv[0].timestamp
            fetched_until = ohlcv[-1].timestamp
            if verbose: print(f"{symbol} fetched: {milliseconds_to_datestring(fetched_since)} to {milliseconds_to_datestring(fetched_until)}")
            if fetched_since >= until_timestamp:# or fetched_since is None:
                print(f"BREAK: fetched_since ({milliseconds_to_datestring(fetched_since)}) >= until_timestamp ({milliseconds_to_datestring(until_timestamp)})")
                break
            all_ohlcv = all_ohlcv + ohlcv
            total_fetched_since = all_ohlcv[0].timestamp
            total_fetched_until = all_ohlcv[-1].timestamp
            
            if verbose: print(f"---> {len(all_ohlcv)} | {milliseconds_to_datestring(total_fetched_since)} to {milliseconds_to_datestring(total_fetched_until)}")
            if fetch_since == total_fetched_until:
                print(f"BREAK: fetch_since ({milliseconds_to_datestring(fetch_since)}) == total_fetched_until ({milliseconds_to_datestring(total_fetched_until)})")
                break
            else:
                fetch_since = total_fetched_until
                time.sleep(rate_limit_buffer_ms)
            # Convert to pandas dataframe and sort

        df = pd.DataFrame(all_ohlcv)
        df.columns = ["open", "high","low","close","volume", "vwap","close_time", "x", "xx"]
        df=df[["open", "high","low","close","volume", "vwap","close_time"]]
        df.index = pd.to_datetime(df["close_time"],unit='ms')
        df.index.name = "datetime"
        df = df.sort_index()
        df.drop_duplicates(subset=["close_time"], keep="last", inplace=True)    
        if verbose: print(f"\nDATA CALLED: {df.index[0]} to {df.index[-1]}, {len(df)} rows\n")
        return df
    
    def load_ohlcvs(self,
                    instruments, 
                    timeframes,
                    since = "2023-01-01 00:00:00",
                    update=False,
                    until = None,
                    limit = 5000,
                    max_retries=100,
                    verbose=True
                    ):
        
        """
        instruments: list of symbols i.e; ['C:AUDUSD','C:EURUSD']
        client instantiation should be done here on demand instead of at class init
        """
        # load from database_path for pickles
        # Example of instruments: "polygon_io__C:AUDUSD"
        instruments_dict = {}
        for instrument in instruments:
            temp = {}
            exchange = instrument.split("__")[0]
            symbol = instrument.split("__")[-1]
            if verbose: print(f"\nGetting <<{symbol}>> from <<{exchange}>> ")
            for timeframe in timeframes:
                if verbose: print(f"{'='*20}\n{symbol} ({timeframe})\n{'='*20}")
                t0 = time.time()
                try:
                    df_old = pickle_helper.pickle_this(data=None, pickle_name=f"{instrument}_{timeframe}",path=self.database_path)
                    if not update:
                        temp[timeframe] = df_old
                        break
                    t1 = np.round(time.time() - t0,2)
                    df_old_end_datetime = df_old.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    df_old_start_datetime = df_old.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    # 
                    
                    if since < df_old_start_datetime:
                        if verbose: print(f"Requested ({since}) < available ({df_old_start_datetime}) ---> Querying from {since}")
                        new_since = since
                    else:
                        if verbose: print(f"available till is outdated ({df_old_end_datetime})")
                        new_since = df_old_end_datetime
                    if verbose: print(f"--> in database ({len(df_old)} rows): {df_old.index[0]} ===> {new_since} ({t1}s)") 
                    
                except Exception as e:
                    print(f"{e}\n {symbol} does not exist in database")
                    df_old=None 
                    new_since = since
                
                
                t0 = time.time()
                
    
                # client = getattr(ccxt, exchange)({'enableRateLimit': True})
                df_updated = self.get_ohlcvs_paginated(symbol = symbol, 
                                                       timeframe = timeframe,
                                                       since = new_since,
                                                       limit = limit, 
                                                       max_retries = max_retries,
                                                       verbose = verbose)
                t1 = np.round(time.time() - t0,2)
                if verbose: print(f"--> queried     ({len(df_updated)} rows): {df_updated.index[0]} ===> {df_updated.index[-1]} ({t1}s)")  
                
                # Merge old database-based df with new updated df
                if df_old is not None:
                    t0 = time.time()
                    df = pd.concat([df_old,df_updated])
                else:
                    df = df_updated.copy()
                df.sort_index(inplace=True)    
                df.drop_duplicates(subset=["close_time"], keep="last", inplace=True)    
                # Save updated df to pickle
                pickle_helper.pickle_this(data=df, pickle_name=f"{instrument}_{timeframe}",path=self.database_path)
                t1 = np.round(time.time() - t0,2)
                if verbose: print(f"--> updated     ({len(df)} rows): {df.index[0]} ===> {df.index[-1]} ({t1}s)") 
                
                temp[timeframe] = df
            instruments_dict[instrument]=temp
            
        return instruments_dict

if __name__ == "__main__":
    #%%
    date_string = "2022-04-25 15:30:00"
    milliseconds = string_to_milliseconds(date_string)
    print(milliseconds)

# %%
