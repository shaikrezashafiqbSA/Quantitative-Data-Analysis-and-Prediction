from datetime import datetime
import pandas as pd
import numpy as np
import time

import ccxt
from utils import pickle_helper
import copy

class KlinesManagerCCXT:
    def __init__(self, 
                 database_path = "./database/klines/"):
        self.database_path = database_path

    def get_ohlcv(self, 
                  client,
                  symbol,
                  timeframe,
                  since, 
                  limit,
                  max_retries=200, 
                  rate_limit_buffer_ms = 3000):
        
        num_retries = 0
        sleep_length = 10000
        while num_retries < max_retries:
            try:
                ohlcv = client.fetch_ohlcv(symbol, timeframe, since, limit)
                client.sleep(rate_limit_buffer_ms)
            except ccxt.BadRequest as e:
                print(e)
                break
            except Exception as e:
                num_retries += 1
                print(f"{e} \n---> retry attempt: {num_retries}")
                client.sleep(sleep_length)
                sleep_length+=10000
            else:
                return ohlcv
    
    def get_ohlcvs_paginated(self,
                             client,
                             symbol = "US500",
                             timeframe = "1h",
                             since = "2022-12-01 00:00:00",
                             limit = 1000,
                             max_retries=300,
                             verbose = True,
                             rate_limit_buffer_ms = 3000,
                             rate_limit_on_no_update = 3000,):
        
        # convert since from string to milliseconds integer if needed
        print(f"===>> since: {since}")
        limit_original = copy.deepcopy(limit)
        limit_new = copy.deepcopy(limit)
        if isinstance(since, str):
            since = client.parse8601(since)
        # preload all markets from the exchange
        # markets = client.load_markets()
        
        # timestamp now
        until_timestamp = client.milliseconds() 
        
        # timeframe (eg: 1h) to timestamp delta
        timeframe_duration_in_seconds = client.parse_timeframe(timeframe)
        timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
        
        # timedelta to paginate by 
        # limit to low limit rest call if since and 
        timedelta = limit * timeframe_duration_in_ms
        first_run = True
        all_ohlcv = []
        while True:
            # Fetch walk backwards (in forward fashion) from until_timestamp
            fetch_since = until_timestamp - timedelta
            if verbose: print(f"\n\n------------------------------------- {limit} -------------------------------------\n")
            if verbose: print(f">          fetch_since = until_timestamp - timedelta\n>          {fetch_since} =  {until_timestamp} - {timedelta}")
            ohlcv = self.get_ohlcv(client, symbol, timeframe, fetch_since, limit, max_retries)
            ohlcv = sorted(ohlcv, key=lambda x: x[0])
            if verbose: print(f">  ===================   ohlcv {len(ohlcv)} rows fetched ===================\n                 {ohlcv}")
            if ohlcv is None:
                print(f"!!  fetched since <<{client.iso8601(fetch_since).replace('T',' ').replace('.000Z','')}>> but no data returned\n                 {ohlcv}")
                client.sleep(rate_limit_on_no_update)
                limit = int(limit_new/2)
                continue
            if len(ohlcv) == 0:
                print(f"!!  fetched since <<{client.iso8601(fetch_since).replace('T',' ').replace('.000Z','')}>> but no data returned\n                 {ohlcv}")
                client.sleep(rate_limit_on_no_update)
                limit = int(limit_new/2)
                continue
            ohlcv = sorted(ohlcv, key=lambda x: x[0])
            fetched_since = ohlcv[0][0]
            fetched_until = ohlcv[-1][0]

            # ================================================================
            # Non Empty data received
            # ================================================================
            print(f"since ---> {since}")
            if verbose: print(f"->                since <<{client.iso8601(since).replace('T',' ').replace('.000Z','')}>>")
            if verbose: print(f"-->         fetch_since <<{client.iso8601(fetch_since).replace('T',' ').replace('.000Z','')}>> until <<{client.iso8601(until_timestamp).replace('T',' ').replace('.000Z','')}>>")
            if verbose: print(f"--->      fetched_since <<{client.iso8601(fetched_since).replace('T',' ').replace('.000Z','')}>> until <<{client.iso8601(fetched_until).replace('T',' ').replace('.000Z','')}>>")
            if fetched_since >= fetched_until:# or fetched_since is None:
                if verbose: print(f"----> !!  fetched since <<{client.iso8601(fetched_since).replace('T',' ').replace('.000Z','')}>> >= until <<{client.iso8601(fetched_until).replace('T',' ').replace('.000Z','')}>>\n                 {ohlcv}")
                # break
                # client.sleep(rate_limit_on_no_update)
                
                until_timestamp = fetched_until - timedelta
                if verbose: print(f"----->            until_timestamp = fetched_since - timedelta\n----->            {until_timestamp} = {fetched_until} - {timedelta}")
                total_fetched_since = all_ohlcv[0][0]
                limit = int(limit_new/2)
                total_fetched_until = until_timestamp
            else:
                limit=limit_original
                until_timestamp = fetched_since
                all_ohlcv = ohlcv + all_ohlcv
                all_ohlcv = sorted(all_ohlcv, key=lambda x: x[0])
                total_fetched_since = all_ohlcv[0][0]
                total_fetched_until = all_ohlcv[-1][0]
            if verbose: print(f"--->  ===================   Total ohlcv {len(all_ohlcv)} rows fetched ===================")

            
            if verbose: print(f"----> total_fetchsince <<{client.iso8601(total_fetched_since).replace('T',' ').replace('.000Z','')}>> until <<{client.iso8601(total_fetched_until).replace('T',' ').replace('.000Z','')}>>")
            if verbose: print(f"----->     fetch_since <<{client.iso8601(fetch_since).replace('T',' ').replace('.000Z','')}>> > since <<{client.iso8601(since).replace('T',' ').replace('.000Z','')}>>\n")
            if fetch_since < since:
                break
            else:
                client.sleep(rate_limit_buffer_ms)
            
        # Convert to pandas dataframe and sort
        df = pd.DataFrame(all_ohlcv)
        df.columns = ["close_time","open", "high","low","close","volume"]
        df=df[["open", "high","low","close","volume","close_time"]]
        df["close_time"]=df["close_time"]/1000
        df.index = pd.to_datetime(df["close_time"],unit='s')
        df.index.name = "datetime"
        df = df.sort_index()
        
        
        return df
    
    
    
    def load_ohlcvs(self,
                    instruments, 
                    timeframes,
                    since = "2022-12-01 00:00:00",
                    update=True,
                    until = None,
                    limit = 1000,
                    max_retries=100,
                    verbose=True
                    ):
        
        """
        instruments: list of symbols i.e; ['US500','ETHUSD']
        client instantiation should be done here on demand instead of at class init
        """
        # load from database_path for pickles
        instruments_dict = {}
        for instrument in instruments:
            temp = {}
            ccxt_exchange = instrument.split("__")[0]
            exchange = ccxt_exchange.split("_")[-1]
            symbol = instrument.split("__")[-1]
            client = getattr(ccxt, exchange)({'enableRateLimit': True})
            if verbose: print(f"\nGetting <<{symbol}>> from <<{exchange}>> ")
            for timeframe in timeframes:
                if verbose: print(f"{'='*20}\n{symbol} ({timeframe})\n{'='*20}")
                t0 = time.time()
                try:
                    df_old = pickle_helper.pickle_this(data=None, pickle_name=f"{symbol}_{timeframe}",path=f"{self.database_path}{ccxt_exchange}/")
                    if not update:
                        temp[timeframe] = df_old
                        break
                    t1 = np.round(time.time() - t0,2)
                    df_old_end_datetime = df_old.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    df_old_start_datetime = df_old.index[0].strftime("%Y-%m-%d %H:%M:%S")
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
                df_updated = self.get_ohlcvs_paginated(client = client,
                                                       symbol = symbol, 
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
                    
                df.drop_duplicates(subset=["close_time"], keep="last", inplace=True)    
                # Save updated df to pickle
                pickle_helper.pickle_this(data=df, pickle_name=f"{instrument}_{timeframe}",path=f"{self.database_path}{ccxt_exchange}/")
                t1 = np.round(time.time() - t0,2)
                if verbose: print(f"--> updated     ({len(df)} rows): {df.index[0]} ===> {df.index[-1]} ({t1}s)") 
                
                temp[timeframe] = df
            instruments_dict[instrument]=temp
            
        return instruments_dict
                
    
    

    
