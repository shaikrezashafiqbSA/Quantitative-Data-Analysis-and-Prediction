import pandas as pd
from utils import pickle_helper
from os import listdir

def csv_concatenator(instruments=["ESUSD"],
                     data_path="./database/instruments/CSVs/",
                     output_path = "./database/instruments/",
                     pickle=True,
                     convert_to_usd_quote=True,
                     fix_tz={"from":"US/Central", "to":"UTC"}):
    
    output_dfs = {}
    for instrument in instruments:
        target_folder = f"{data_path}{instrument}"
        year_folders = listdir(target_folder)
        year_folders = list(filter(lambda f: f. startswith('20'), year_folders))
        print(f"{'='*40}\n{instrument} --> {year_folders} \n{'='*40}")
        col_rename = {"Open":"open",
                      "High":"high",
                      "Low":"low",
                      "Last":"close",
                      "Volume":"volume",
                      "Time":"date_time"}


        full_list_of_dfs = []
        for year_folder in year_folders:
            # print(year_folder)
            all_files = listdir(f"{target_folder}/{year_folder}")
            # print(all_files)
            csv_files = list(filter(lambda f: f. endswith('.csv'), all_files))

            list_of_dfs = []
            for csv_file in csv_files:
                df_i=pd.read_csv(f"{target_folder}/{year_folder}/{csv_file}")[:-1]
                list_of_dfs.append(df_i)

            df_year=pd.concat(list_of_dfs)


            df_year.rename(columns=col_rename,inplace=True)
            df_year = df_year[col_rename.values()]
            
                
            df_year["date_time"]=pd.to_datetime(df_year["date_time"])
            df_year.set_index("date_time",inplace=True)
            df_year.sort_index(inplace=True)
            
            if fix_tz is not None:
                df_year.index = df_year.index.tz_localize(fix_tz["from"])#.tz_convert('UTC')
                df_year.index = df_year.index.tz_convert(fix_tz["to"])
                df_year.index = df_year.index.tz_convert(None)
                
                
            full_list_of_dfs.append(df_year)
            print(f"{year_folder} collated: {df_year.index[0]} to {df_year.index[-1]}") 
        final_df = pd.concat(full_list_of_dfs)
        final_df.sort_index(inplace=True)
        final_df["close_time"]=final_df.index.astype(int)// (10 ** 9)
        print(f"\n{instrument} JOB DONE\n TOTAL Collated: {final_df.index[0]} to {final_df.index[-1]}\n TOTAL ROWS:{len(final_df)}\n\n")
        
        # If any not USD denominated (eg; USDTHB) to USD denominated (eg; THBUSD)
        if instrument.split("_")[1] != "USD" and convert_to_usd_quote:
            base = instrument.split("_")[1] 
            final_df[["open","high","low","close"]]=1/final_df[["open","high","low","close"]]
            target_folder = f"{data_path}{base}_USD"
            
            if pickle: pickle_helper.pickle_this(final_df, pickle_name=f"{base}_USD", path=output_path)  
            output_dfs[f"{base}_USD"]=final_df
        else:    
            if pickle: pickle_helper.pickle_this(final_df, pickle_name=f"{instrument}", path=output_path)
            output_dfs[instrument] = final_df
    return output_dfs 


import warnings
def xls_concatenator(instrument = "AUDX_USD",
                     path = f"./database/instruments/CSVs/"):

    path = f"{path}/{instrument}/"
    
    print(f" path: {path}")
    all_files = sorted(listdir(path))

    df = None
    for file_name in all_files:#[2:4]:
        print(f"Reading ---> {file_name}")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df_i=pd.read_excel(f"{path}/{file_name}", header=None, index_col=0,engine="openpyxl")
        df_i.index.name = "date_time"
        df_i.columns = ["open","high","low","close","volume"]
        print(f"             {df_i.index[0]} --> {df_i.index[-1]} ==========> len: {len(df_i)}")

        if df is None:
            df = df_i.copy()
        else:
            df = pd.concat([df,df_i])#left_index = True, right_index=True, how="outer")
        print(f"             Merged {df.index[0]} --> {df.index[-1]} ==========> len: {len(df)}")



    fix_tz = {"from":"EST","to":"UTC"}
    df.index = df.index.tz_localize(fix_tz["from"])#.tz_convert('UTC')
    df.index = df.index.tz_convert(fix_tz["to"])
    df.index = df.index.tz_convert(None)

    df["close_time"]=df.index.astype(int)// (10 ** 9)

    return df

if __name__ == "__main__":
    #%%
    from os import listdir
    path = f"/Users/shaik/PycharmProjects/trading_signal_V2/database/histdata.com/AUDUSD/"
    all_files = sorted(listdir(path))
    # %%
    df = xls_concatenator(instrument="AUDUSD", path="/Users/shaik/PycharmProjects/trading_signal_V2/database/histdata.com/")
# %%
