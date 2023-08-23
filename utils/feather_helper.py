
import pandas as pd

contract_month_codes = {"F":1,
                        "G":2,
                        "H":3,
                        "J":4,
                        "K":5,
                        "M":6,
                        "N":7,
                        "Q":8,
                        "U":9,
                        "V":10,
                        "X":11,
                        "Z":12}

if __name__ == "__main__":
    file_name = "USDF_TICK_joined_2019-09-04_2022-09-07.fea"
    file_name = "USDTHB_FUT_combined.fea"
    df0 = pd.read_feather(path=f"./database/{file_name}")
    df0.set_index("index", inplace=True)
    
    contracts = [i.split("_")[0] for i in list(df0.filter(regex="MDAsk1Price_close").columns)]
    # df = df0.filter(regex="USDU19").filter(regex="Price_close")
    # df = df.filter(regex="close")
    # df = df.filter(regex="Ask1Price")
    
    # df1 = df.head(10)
    # df.rename(columns={"index":"date_time"},inplace=True)
    # df.index = pd.to_datetime(df["date_time"])
    # df.drop(columns=["date_time"],inplace=True)
    # contracts = df.filter(regex="LastPrice_close").columns
    # contracts = ["_".join(contract.split("_")[:-2]) for contract in contracts]
    # print(contracts)
