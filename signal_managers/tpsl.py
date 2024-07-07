
import numpy as np



def calc_signal_tpsl(df0, 
                    penalty = 1, # this widens the SL so that it is not hit too often
                    tp_position_dict = {"TP1": {"L":{"lookback":3, "qtl": 0.3}, 
                                                "S": {"lookback":3, "qtl":0.3}
                                                },
                                        "TP2": {"L":{"lookback":6, "qtl": 0.6}, 
                                                "S": {"lookback":6, "qtl":0.6}
                                                },
                                        "TP3": {"L":{"lookback":9, "qtl": 0.9}, 
                                                "S": {"lookback":9, "qtl":0.9}
                                                }
                                        }
                    ):
    df = df0.copy()
    # create a column to count the duration of each trade
    for position in ["L", "S"]:
        # Create a column to count the duration of each trade
        df[f'{position}_dur'] = df.groupby(df[f'{position}_id']).cumcount()
        df[f'{position}_dur'] = np.where(df[f'{position}_id'].isna(), np.nan, df[f'{position}_dur'])

        # Get winning trades PNL==strength AND duration==t
        df[f"{position}_strength"] = np.where((df[f"{position}_rpnl"]>0), df[f"{position}_rpnl"]/df[f"{position}_qty"], np.nan)
        df[f"{position}_strength_t"] = np.where((df[f"{position}_rpnl"]>0), df[f'{position}_dur'], np.nan)
        # get losing trades PNL==weakness AND duration==t
        df[f"{position}_weakness"] = np.where((df[f"{position}_rpnl"]<0), df[f"{position}_rpnl"]/df[f"{position}_qty"], np.nan)
        df[f"{position}_weakness_t"] = np.where((df[f"{position}_rpnl"]<0), df[f'{position}_dur'], np.nan)

    for tp in ["TP1", "TP2", "TP3"]:
        for position in ["L", "S"]:
            # print(f"{tp} --> {position}")
            try:
                lookback = tp_position_dict[tp][position]["lookback"]
            except Exception as e:
                print(f"tp_position_dict[{tp}][{position}] does not exist: {e}")
                continue
            qtl = tp_position_dict[tp][position]["qtl"]
            df[f"{position}_{tp}_strength"] = df[f"{position}_strength"].dropna().rolling(lookback).quantile(qtl)
            df[f"{position}_{tp}_strength"]=df[f"{position}_{tp}_strength"].fillna(method="ffill")

            df[f"{position}_{tp}_weakness"] = df[f"{position}_weakness"].dropna().rolling(lookback).quantile(1-qtl)
            df[f"{position}_{tp}_weakness"]=df[f"{position}_{tp}_weakness"].fillna(method="ffill")

            df[f"{position}_{tp}_strength_t"] = df[f"{position}_strength_t"].dropna().rolling(lookback).quantile(qtl)#-1
            df[f"{position}_{tp}_strength_t"]=df[f"{position}_{tp}_strength_t"].fillna(method="ffill")

            df[f"{position}_{tp}_weakness_t"] = df[f"{position}_weakness_t"].dropna().rolling(lookback).quantile(1-qtl)#-1
            df[f"{position}_{tp}_weakness_t"]= df[f"{position}_{tp}_weakness_t"].fillna(method="ffill")

            if position == "L":
                x = 1 
            elif position == "S":
                x = -1

            # PRICE TP and SL
            df[f"{position}_{tp}"] = df["close"] +x*df[f"{position}_{tp}_strength"] 
            df[f"{position}_SL{tp[-1]}"] = df["close"] -x*penalty*abs(df[f"{position}_{tp}_weakness"])

            # TIME TP AND SL
            df[f"{position}_{tp}_t"] = df[f"{position}_{tp}_strength_t"] 
            df[f"{position}_SL{tp[-1]}_t"] = df[f"{position}_{tp}_weakness_t"]

            # Risk reward ratio
            df[f"{position}_RR{tp[-1]}"] = abs(df["close"]-df[f"{position}_SL{tp[-1]}"]) / abs(df["close"] - df[f"{position}_{tp}"])
            df[f"{position}_FP{tp[-1]}"] = (df[f"{position}_SL{tp[-1]}"] + df[f"{position}_{tp}"])/2
    
    return df



def calc_signal_tpsl(df0, 
                    penalty = 1, # this widens the SL so that it is not hit too often
                    tp_position_dict = {"TP1": {"L":{"lookback":3, "qtl": 0.3}, 
                                                "S": {"lookback":3, "qtl":0.3}
                                                },
                                        "TP2": {"L":{"lookback":6, "qtl": 0.6}, 
                                                "S": {"lookback":6, "qtl":0.6}
                                                },
                                        "TP3": {"L":{"lookback":9, "qtl": 0.9}, 
                                                "S": {"lookback":9, "qtl":0.9}
                                                }
                                        }
                    ):
    df = df0.copy()
    # create a column to count the duration of each trade
    for position in ["L", "S"]:
        # Create a column to count the duration of each trade
        df[f'{position}_dur'] = df.groupby(df[f'{position}_id']).cumcount()
        df[f'{position}_dur'] = np.where(df[f'{position}_id'].isna(), np.nan, df[f'{position}_dur'])

        # Get winning trades PNL==strength AND duration==t
        df[f"{position}_strength"] = np.where((df[f"{position}_rpnl"]>0), df[f"{position}_rpnl"]/df[f"{position}_qty"], np.nan)
        df[f"{position}_strength_t"] = np.where((df[f"{position}_rpnl"]>0), df[f'{position}_dur'], np.nan)
        # get losing trades PNL==weakness AND duration==t
        df[f"{position}_weakness"] = np.where((df[f"{position}_rpnl"]<0), df[f"{position}_rpnl"]/df[f"{position}_qty"], np.nan)
        df[f"{position}_weakness_t"] = np.where((df[f"{position}_rpnl"]<0), df[f'{position}_dur'], np.nan)

    for tp in ["TP1", "TP2", "TP3"]:
        for position in ["L", "S"]:
            # print(f"{tp} --> {position}")
            try:
                lookback = tp_position_dict[tp][position]["lookback"]
            except Exception as e:
                print(f"tp_position_dict[{tp}][{position}] does not exist: {e}")
                continue
            qtl = tp_position_dict[tp][position]["qtl"]
            df[f"{position}_{tp}_strength"] = df[f"{position}_strength"].dropna().rolling(lookback).quantile(qtl)
            df[f"{position}_{tp}_strength"]=df[f"{position}_{tp}_strength"].fillna(method="ffill")

            df[f"{position}_{tp}_weakness"] = df[f"{position}_weakness"].dropna().rolling(lookback).quantile(1-qtl)
            df[f"{position}_{tp}_weakness"]=df[f"{position}_{tp}_weakness"].fillna(method="ffill")

            df[f"{position}_{tp}_strength_t"] = df[f"{position}_strength_t"].dropna().rolling(lookback).quantile(qtl)#-1
            df[f"{position}_{tp}_strength_t"]=df[f"{position}_{tp}_strength_t"].fillna(method="ffill")

            df[f"{position}_{tp}_weakness_t"] = df[f"{position}_weakness_t"].dropna().rolling(lookback).quantile(1-qtl)#-1
            df[f"{position}_{tp}_weakness_t"]= df[f"{position}_{tp}_weakness_t"].fillna(method="ffill")

            if position == "L":
                x = 1 
            elif position == "S":
                x = -1

            # PRICE TP and SL
            df[f"{position}_{tp}"] = df["close"] +x*df[f"{position}_{tp}_strength"] 
            df[f"{position}_SL{tp[-1]}"] = df["close"] -x*penalty*abs(df[f"{position}_{tp}_weakness"])

            # TIME TP AND SL
            df[f"{position}_{tp}_t"] = df[f"{position}_{tp}_strength_t"] 
            df[f"{position}_SL{tp[-1]}_t"] = df[f"{position}_{tp}_weakness_t"]

            # Risk reward ratio
            df[f"{position}_RR{tp[-1]}"] = abs(df["close"]-df[f"{position}_SL{tp[-1]}"]) / abs(df["close"] - df[f"{position}_{tp}"])
            df[f"{position}_FP{tp[-1]}"] = (df[f"{position}_SL{tp[-1]}"] + df[f"{position}_{tp}"])/2
    
    return df