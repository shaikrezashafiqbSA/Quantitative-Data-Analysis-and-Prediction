
import pandas as pd
import numpy as np
import time

from tvDatafeed import TvDatafeed, Interval
from utils import pickle_helper
import asyncio

class KlinesManagerTV:
    def __init__(self, 
                 username:str = "shaikrezashafiq",
                 password:str = "SRS271828!",
                 database_path = "./database/TVdata/"): #./database/scalers/
            
        self.client = TvDatafeed(username,password)
        self.database_path = database_path
        self.klines_freq_dict = {"1m":Interval.in_1_minute,
                                 "3m":Interval.in_3_minute,
                                 "5m":Interval.in_5_minute,
                                 "15m":Interval.in_15_minute,
                                 "30m":Interval.in_30_minute,
                                 "45m":Interval.in_45_minute,
                                 "1h":Interval.in_1_hour,
                                 "2h":Interval.in_2_hour,
                                 "3h":Interval.in_3_hour,
                                 "4h":Interval.in_4_hour,
                                 "1d":Interval.in_daily,
                                 "1w":Interval.in_weekly,
                                 "1M":Interval.in_monthly}
        
        
    def query_symbol(self,instrument, exchange=""):
        result = pd.DataFrame(self.client.search_symbol(instrument,exchange))
        return result
        
        
    def get_ohlcv(self,
                  exchange, 
                  symbol,
                  timeframe, 
                  limit = 20000,
                  max_retries = 3,
                  timezone = {"from":"Asia/Singapore", "to":"UTC"},
                  verbose=False,
                  ):
        """
        Returns klines dataframe with 
        index:
            datetime
        columns:
            open
            high
            low
            close
            volume
            close_time
            
        """
        
        interval = self.klines_freq_dict[timeframe]
        
        # LOAD TVDATAFEED        
        tries = 0
        while tries < max_retries:
            try:
                klines = self.client.get_hist(symbol=symbol,
                                          exchange=exchange,
                                          interval=interval,
                                          n_bars=limit)
            except Exception as e:
                print(e)
                print(f"RETRYING ... {tries}")
                time.sleep(10)
                tries+=1
            else:
                klines.drop(columns=["symbol"],inplace=True)
                klines = klines[["open","high","low","close","volume"]]
                if timezone is not None:
                    klines.index = klines.index.tz_localize(timezone["from"])#.tz_convert('UTC')
                    klines.index = klines.index.tz_convert(timezone["to"])
                    klines.index = klines.index.tz_convert(None)
                    klines["close_time"]=klines.index.astype(int)//(10 ** 9)
                    
                return klines
                break
            
        return klines
    
    def load_ohlcvs(self,
                    instruments, 
                    timeframes,
                    limit = 10000,
                    timezone = {"from":"Asia/Singapore", "to":"UTC"},
                    max_retries=3,
                    verbose=True
                    ):
        # load from database_path for pickles
        instruments_dict = {}
        for instrument in instruments:
            temp = {}
            exchange,symbol = instrument.split("__")
            if verbose: print(f"\nGetting <<{symbol}>> from <<{exchange}>> ... ")
            for timeframe in timeframes:
                if verbose: print(f"{'='*20}\n{instrument.split('__')[-1]} ({timeframe})\n{'='*20}")
                t0 = time.time()
                try:
                    df_old = pickle_helper.pickle_this(data=None, pickle_name=f"{instrument}_{timeframe}",path=self.database_path)
                    t1 = np.round(time.time() - t0,2)
                    if verbose: print(f"--> in database ({len(df_old)} rows): {df_old.index[0]} ===> {df_old.index[-1]} ({t1}s)") 
                except Exception as e:
                    print(e)
                    df_old=None 
                
                
                t0 = time.time()
                df_updated = self.get_ohlcv(exchange=exchange,
                                            symbol=symbol, 
                                            timeframe=timeframe,
                                            limit=limit, 
                                            max_retries = max_retries,
                                            timezone = timezone,
                                            verbose=verbose)
                t1 = np.round(time.time() - t0,2)
                if verbose: print(f"--> queried     ({len(df_updated)} rows): {df_updated.index[0]} ===> {df_updated.index[-1]} ({t1}s)")  
                
                # Merge old database-based df with new updated df
                if df_old is not None:
                    t0 = time.time()
                    df = pd.concat([df_old,df_updated])
                    df.drop_duplicates(subset=["close_time"], keep="last", inplace=True)
                else:
                    df = df_updated.copy()
                # Save updated df to pickle
                pickle_helper.pickle_this(data=df, pickle_name=f"{instrument}_{timeframe}",path=self.database_path)
                t1 = np.round(time.time() - t0,2)
                if verbose: print(f"--> updated     ({len(df)} rows): {df.index[0]} ===> {df.index[-1]} ({t1}s)") 
                
                temp[timeframe] = df
            instruments_dict[instrument]=temp
            
        return instruments_dict
                
    def get_ohlcvs(self,
                   instruments,
                   timeframes, 
                   limit = 10000,
                   timezone = {"from":"Asia/Singapore", "to":"UTC"},
                   max_retries=3,
                   verbose=True
                   ):
        
        
        instruments_dict = {}
        for instrument in instruments:
            temp = {}
            exchange,symbol = instrument.split("__")
            if verbose: print(f"Getting <<{symbol}>> from <<{exchange}>> ... ")
            for timeframe in timeframes:
        
                df = self.get_ohlcv(exchange=exchange,
                                    symbol=symbol, 
                                    timeframe=timeframe,
                                    limit=limit, 
                                    max_retries = max_retries,
                                    timezone = timezone,
                                    verbose=verbose)
                
                    
                if verbose: print(f"--> {instrument.split('__')[-1]} ({timeframe})\n--> Loaded ({len(df)} rows): {df.index[0]} ===> {df.index[-1]}\n")    
                temp[timeframe] = df
                
            # instrument_renamed = instrument.split("__")[-1]
            # instruments_dict[instrument_renamed] = temp
            
            instruments_dict[instrument] = temp
        
        return instruments_dict
    
    async def get_ohlcv_async(self,
                              exchange, 
                              symbol,
                              timeframe, 
                              limit = 2,
                              max_retries = 3,
                              timezone = {"from":"Asia/Singapore", "to":"UTC"},
                              verbose=False,
                              include_timestamp = True,
                              ):
        
        t0 = time.time()
        print(f"LOADING {symbol} from {exchange} ...")
        df = self.get_ohlcv(exchange=exchange,
                            symbol=symbol, 
                            timeframe=timeframe,
                            limit=limit, 
                            max_retries = max_retries,
                            timezone = timezone,
                            verbose=verbose)
        await asyncio.sleep(0)
        t1 = round(time.time()-t0,3)
        print(f"LOADED {symbol} from {exchange} {t1}s")
        if include_timestamp:
            df["timestamp"]=t0*1e9
        return df
        
        
    async def get_ohlcvs_async(self,
                   instruments,
                   timeframes, 
                   limit = 2,
                   timezone = {"from":"Asia/Singapore", "to":"UTC"},
                   max_retries=3,
                   verbose=True
                   ):
        
        
        tasks = []
        instruments_dict = {}
        for instrument in instruments:
            exchange,symbol = instrument.split("__")
            task = asyncio.gather(*[self.get_ohlcv_async(exchange=exchange,
                                symbol=symbol, 
                                timeframe=timeframe,
                                limit=limit, 
                                max_retries = max_retries,
                                timezone = timezone,
                                verbose=verbose) for timeframe in timeframes])
            tasks.append(task)
        print(f"\n{'_'*100}\nGathering tasks\n{'_'*100}")
        res = await asyncio.gather(*tasks) # this is a list of dataframes, need to reorganize to nest dict
        print(f"\n{'_'*100}\nGathering done\n{'_'*100}")
        for i,instrument in enumerate(instruments):
            temp = {}
            for j, timeframe in enumerate(timeframes):
                temp[timeframe] = res[i][j]
            instruments_dict[instrument] = temp
            
        return instruments_dict
        
        
        
#%%
if __name__ == "__main__":
    #%%
    ktv = KlinesManagerTV()
    instrument = "EURUSD"
    check = ktv.query_symbol(instrument=instrument)
    check = check[check["type"]=="forex"]
        
    
    #%%
    tv = KlinesManagerTV()
    
    instruments = ["OANDA__USDSGD", "OANDA__EURUSD"]
    timeframes = ["1m"]
    # test = tv.get_ohlcvs1(instruments = instruments, timeframes=timeframes)
    t0=time.time()
    res = asyncio.run(tv.get_ohlcvs_async(instruments = instruments, timeframes=timeframes))
    t1= round(time.time()-t0,3)
    print(f"time taken: {t1} s")
    print(res)
    # loop = asyncio.get_event_loop()
    #
    # res = loop.create_task(tv.get_ohlcvs1(instruments = instruments, timeframes=timeframes))
