import pandas as pd
import os
from datetime import datetime
import pytz
from utils.pickle_helper import pickle_this

def load_ohlcv(update=False, path = "./database/klines/barchart/USDSGD_1m"):
    df = pickle_this(data=None, pickle_name = "",path=path)
    if (df is None) or update:
        # specify the folder
        csv_folder = "./database/klines/barchart/USDSGD_1m_raw"

        # get a list of all csv files in the folder
        csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

        # initialize a list to store dataframes
        dfs = []

        # loop through the csv files and append to the dataframe list
        for file in csv_files: 
            data = pd.read_csv(os.path.join(csv_folder, file))
            # remove the last row of each csv file
            data = data.iloc[:-1]
            dfs.append(data)

        # concatenate all dataframes in the list
        df = pd.concat(dfs, ignore_index=True)

        # convert 'Time' column to datetime
        df['datetime'] = pd.to_datetime(df['Time'])
        # convert time to UTC
        df['datetime'] = df['datetime'].dt.tz_localize("US/Central").dt.tz_convert('UTC')
        # set 'Time' column as index
        df.set_index('datetime', inplace=True)

        # create 'close_time' column as unix timestamp
        # df['close_time'] = (df.index - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1ms')
        df['close_time'] =  df.index.astype('int64') // 1e6
        df['close_time'] = df['close_time'] .astype('int64')

        # sort the dataframe
        df.sort_index(inplace=True)
        # keep only ['open', 'high', 'low', 'close', 'volume', 'close_time'] columns
        # select only the desired columns
        df = df[['Open', 'High', 'Low', 'Last', 'Volume', 'close_time']]

        # rename the 'Last' column to 'close'
        df.rename(columns={'Last': 'close'}, inplace=True)
        df.columns = [col.lower() for col in df.columns]
        df.index = df.index.tz_localize(None)

        pickle_this(data=df, pickle_name = "",path=path)
    return df