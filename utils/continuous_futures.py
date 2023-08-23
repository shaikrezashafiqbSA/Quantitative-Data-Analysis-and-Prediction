# pip install pyarrow
import pandas as pd



def reformat_ohlcv(df):
    df.columns = ["close","high","low","open", "number_of_trades","volume"]
    df = df[["open","high","low","close", "number_of_trades","volume"]]
    # Fill na volume/ no_of_trades with 0
    df[["number_of_trades","volume"]] = df[["number_of_trades","volume"]].fillna(0)
    # Fill no trade bars with previous close info
    # df[["close"]] = df[["close"]].fillna(method="ffill")
    # # sideward fill ohl with close
    # for col in ['open', 'high', 'low']:
    #     df[col] = df[col].fillna(df['close'])
    return df


def read_futures_fea(file_name):
    """
    This function assumes contracts are concatenated in increasing order along dataframe columns
    
    """
    df = pd.read_feather(path=f"./database/{file_name}")
    df.rename(columns={"index":"date_time"},inplace=True)
    df.index = pd.to_datetime(df["date_time"])
    df.drop(columns=["date_time"],inplace=True)
    
    contracts = df.filter(regex="LastPrice_close").columns
    contracts = ["_".join(contract.split("_")[:-2]) for contract in contracts]
    # print(contracts)
    futures_dict = {}
    contract_expiries = []
    for contract in contracts:
        # Drop all rows where ALL columns in one contract is NAN --> where volume is nan particularly, means that contract is not tradable
        # NOTE: contracts should end by 0400 on expiry date
        df_i = df.filter(regex=contract).dropna(how="all").copy() 
        # df_i = df_i.resample("5Min").last()
        df_i = reformat_ohlcv(df_i)
        df_i["close_time"]=df_i.index.astype(int)// (10 ** 9)
        
        contract_expiry = contract.split("_")[0]
        contract_expiries.append(contract_expiry)
        
        temp = {"df":df_i, "contract":contract.split("_")[1]}
        futures_dict[contract_expiry] = df_i

        print(f"{contract_expiry} ---> last datetime: {df_i.index[-1]} --> len: {len(df_i)}")

    return futures_dict, contract_expiries, df




import numpy as np
import pandas as pd
from utils import pickle_helper 
def perpetual_rollover(df_front, df_back0, rollover_bars=288, data_cols = ['open', 'low', 'high', 'close', 'number_of_trades', 'volume', 'close_time'], verbose=False):
    df_back=df_back0.copy()
    
    """
    both df_front and df_back starts from expiry of previous contract
    df_front, df_back ends on their own expiries
    
    # this must return df_front since df_front is expanding, df_back is constant size slice  !!! 
    ---> how? must concat df_front and df_back then update 
    
    
    This function will stitch these two using the perpetual method
    And then return updated df_back with continuous prices
    """
    front_expiry_datetime = df_front.index[-1]
    # Ensure back has a month's worth of data from previous contract expiry! else non expired contract (open contract)
    df_back_spliced = df_back[front_expiry_datetime:]
        
    # Construct decay weights matrix
    # eg: 0, 0.1, 0.2, .... 1 for rollover_bars of 10 for all columns (except close time)
    # so len(data_cols 
    decay_weights = np.repeat(np.linspace(0, 1, rollover_bars + 1)[1:], int(len(data_cols)-1)).reshape(rollover_bars, int(len(data_cols)-1))
    
    # Multiply front with decreasing weights eg; 1, 0.9,..., 0
    df_front_rollover = df_front.iloc[-rollover_bars:,:-1]
    df_weighted_front_rollover = df_front_rollover * (1-decay_weights)
    
    # Find rollover start date so that can exactly overlap decay between front and back 
    rollover_start_datetime = df_front_rollover.index[0]
    df_back_rollover = df_back[rollover_start_datetime:].iloc[:rollover_bars,:-1]
    
    # Multiply back with increasing weights eg: 0, 0.1, 0.2,..., 1
    df_weighted_back_rollover = df_back_rollover * decay_weights
    
    # Combine decayed prices together to form perpetual weighted continuous futures prices
    df_perp = df_weighted_front_rollover.add(df_weighted_back_rollover)
    
    # Concatenate spliced (non-continuous) prices
    f_start=df_front.index[0]
    f_end = df_front.index[-1]
    df_front = pd.concat([df_front, df_back_spliced])
    
    # Update spliced (non-continouous) prices with perpetual prices
    df_front.update(df_perp)
    
    if verbose: print(f"--> Extending: {f_start} -> {f_end}  with   \n                                      {df_back_spliced.index[0]} -> {df_back_spliced.index[-1]} \n                         PERP WINDOW: {df_perp.index[0]} -> {df_perp.index[-1]}\n")
    
    return df_front




def calc_perp(futures_dict, 
              rollover_bars=288, 
              verbose=True,
              qtrly=True, 
              set_quote_USD=False,
              to_pickle= True,
              pickle_path = "./database/instruments/",
              pickle_name = "perp"):

    df_t0 = None
    for i, (contract, df) in enumerate(futures_dict.items()):
        quarterly_contract = pd.to_datetime(contract).month%3 == 0
        if qtrly and not quarterly_contract:
            continue
            
        if df_t0 is None:
            if verbose: print(f"STARTING Contract: {contract} \n--> {df.index[0]} -> {df.index[-1]}\n")
            df_t0 = df

        # Need to stop building perp when 
        else:
            # Ensure back contract has more than rollover_bars data
            front_end_datetime = df_t0.index[-1]
            l_back = len(df[front_end_datetime:])
            if l_back > rollover_bars:
                if verbose: print(f"Contract: {contract}")#len: {l_back}")
                df_t0 = perpetual_rollover(df_t0, df, rollover_bars = rollover_bars, verbose=verbose)
            else:
                if verbose: print(f"NOT ENOUGH DATA at {contract} --> len: {l_back} , contract last datetime: {front_end_datetime} ----> SKIP TO NEXT NEAREST CONTRACT\n")
                continue

    if verbose: print(f"\n{'-'*40}\n SUMMARY\n{'-'*40}\n range: {df_t0.index[0]} to {df_t0.index[-1]} \nlen: {len(df_t0)}")
    if set_quote_USD:
        df_t0[["open","high","low","close"]]=1/df_t0[["open","high","low","close"]]
    
    if to_pickle:
        pickle_helper.pickle_this(df_t0, pickle_name=pickle_name, path=pickle_path)  
        
        
    return df_t0