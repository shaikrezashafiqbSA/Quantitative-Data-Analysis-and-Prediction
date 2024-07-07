import pandas as pd
import os
from datetime import datetime
import pytz
from utils.pickle_helper import pickle_this

def load_ohlcvs(instruments = ["USDSGD"],
                timeframes = ["1m"],
                since = "2020-01-01 00:00:00",
                limit = 1000,
                update=False, 
                path = "./database/klines/metatrader/"): #USDSGD_1m"):
    
    instruments_dict = {}
    for instrument in instruments:
        temp = {}
        for timeframe in timeframes:
            print(f"METATRADER LOADING: {instrument}")
            path_instrument_timeframe = f"{path}{instrument}_{timeframe}"
            df = pickle_this(data=None, pickle_name = "",path=path_instrument_timeframe)
            if (df is None) or update:
                # specify the folder
                csv_folder = path_instrument_timeframe + "_raw/"
                # csv_folder = "./database/klines/metatrader/USDSGD_1m_raw/"

                # get a list of all csv files in the folder
                csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
                # print(csv_files)
                # initialize a list to store dataframes
                dfs = []

                # loop through the csv files and append to the dataframe list
                for file in csv_files: 
                    # specify column names
                    column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                    # read csv with column names
                    data = pd.read_csv(os.path.join(csv_folder, file), names=column_names)
                    # print(f"\n{file}: BEFORE\n{data}\n")
                    # combine 'date' and 'time' columns and convert to datetime
                    data["datetime"] = data['date'] + ' ' + data['time']
                    data['datetime'] = pd.to_datetime(data["datetime"] , format='%Y.%m.%d %H:%M') 
                    # drop the original 'date' and 'time' columns
                    data.drop(['date', 'time'], axis=1, inplace=True)
                    # print(f"\n{file}: AFTER\n{data}\n")
                    # drop rows containing many nans
                    data.dropna(thresh=4, inplace=True)
                    dfs.append(data)


                # concatenate all dataframes in the list
                df = pd.concat(dfs, ignore_index=True)
                print(df.columns)
                # convert 'Time' column to datetime
                # convert time to UTC
                df['datetime'] = df['datetime']#.dt.tz_localize("US/Central").dt.tz_convert('UTC')
                # set 'Time' column as index
                df.set_index('datetime', inplace=True)

                # create 'close_time' column as unix timestamp
                # df['close_time'] = (df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1ms')
                df['close_time'] =  df.index.astype('int64') // 1e6
                df['close_time'] = df['close_time'] .astype('int64')

                # sort the dataframe
                df.sort_index(inplace=True)
                # keep only ['open', 'high', 'low', 'close', 'volume', 'close_time'] columns
                df.columns = [col.lower() for col in df.columns]
                df.index = df.index.tz_localize(None)

                pickle_this(data=df, pickle_name = "",path=path_instrument_timeframe)
            temp[timeframe] = df
        instruments_dict[instrument] = temp
    return instruments_dict