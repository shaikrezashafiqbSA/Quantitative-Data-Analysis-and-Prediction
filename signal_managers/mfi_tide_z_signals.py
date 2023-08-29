import numpy as np
import importlib
from signal_managers import indicators
importlib.reload(indicators)
import time

from utils.list_type_converter import convert_to_type,convert_to_array
from signal_managers.indicators import calc_mfi_sig, param_func_mfi_EMAVol, param_func_mfi
from signal_managers.indicators import calc_tide_sig, calc_tide_sig, param_func_tide_EMAVol, param_func_tide
from signal_managers.indicators import calc_z_sig, param_func_Z_EMAVol, param_func_Z,calc_sig_strengths
# chosen_to_debug = ["ccxt_currencycom__DXY", "2h"] 


import pandas as pd
from klines_managers import klines_ccxt
kline_manager = klines_ccxt.KlinesManagerCCXT()
from signal_managers.resampler import calc_klines_resample

def load_klines(instruments, timeframes):
    instruments_dict = kline_manager.load_ohlcvs(instruments = instruments,
                                            timeframes = timeframes,
                                            since = "2018-01-01 00:00:00",
                                            limit = 1000, update=False, verbose=False)

    return instruments_dict


def resample_instruments_dict(instruments_dict,
                              resample_to_list = ["2h", "3h", "4h", "6h", "12h"],
                              first_timeframe = "1h"):
    # Resample from 1hr --> 2h, 3h, 4h, 6h, 12h, 24h, 

    resample_to_list = ["2h", "3h", "4h", "6h", "12h"]#, "24h"]
    first_timeframe = "1h"

    # resample_to_list = ["15m", "20m", "30m", "60m"]
    # first_timeframe = "5m"
    for instrument,instrument_dict in instruments_dict.items():
        # print(f"\n{instrument}\n")
        for resample_to in resample_to_list:
            df = instrument_dict[first_timeframe].copy()
            window = int(int(resample_to[:-1])/int(first_timeframe[:-1]))
            # print(f"Resampling {first_timeframe} to {resample_to} --> window: {window}")
            if "m" in resample_to:
                resample_to_formatted = resample_to[:-1]+"T"
            else:
                resample_to_formatted = resample_to
            df_resampled = calc_klines_resample(df,window=window, resample_to=resample_to_formatted)
            df_resampled.fillna(method="ffill", inplace=True)
            df_resampled.dropna(inplace=True)
            # print(df_resampled.columns)
            instruments_dict[instrument][resample_to] = df_resampled
    return instruments_dict


def calc_mfi_tide_z_signals(instruments_dict, chosen_to_debug = []):
    chosen_to_debug = [] 
    sig_dicts = {}
    for instrument,instrument_dict in instruments_dict.items():
        if (instrument not in chosen_to_debug) and not (len(chosen_to_debug)==0):
            continue
        sig_dict = {}
        for timeframe in instrument_dict.keys():
            # print(f"timeframe: {timeframe} is in {chosen_to_debug}")
            if (timeframe not in chosen_to_debug) and not (len(chosen_to_debug)==0):
                continue
            """
            # SIGNAL CALCULATION FOR 1 df (1 instrument, 1 timeframe) 
            """
            dfm= instruments_dict[instrument][timeframe].copy()
            print(f"---> {instrument} {timeframe} : Initialise feature generation ...")


            start = time.time()
            # print(f"\n{'='*30}\MFI\n{'='*30}")
            dfm = calc_mfi_sig(dfm,
                                cols_set=[['open','high','low']],
                                param_func_mfi= param_func_mfi,
                                dynamic_param_col = ["volume"],
                                fixed_window = True)
            end = time.time()
            # print(f"------> Time taken (MFI): {np.round(end-start,3)} seconds\n")

            start = time.time()
            # print(f"\n{'='*30}\nTIDE\n{'='*30}")
            # CUSTOM PARAM FUNC ================================================
            def param_func_tide_MFI(x,i, col_index = 0):
                q_Lb, q_Ls, q_Sb, q_Ss = 0.9, 0.1, 0.1, 0.9
                # print(f"x[:,col_index[0]]: {x[:,col_index[0]]}, len: {len(x[:i+1,col_index[0]])}")
                # mfi = x[i+1-14:i+1,col_index[0]]
                
                mfi_np = x[:i+1,col_index[0]]
                # print(f"{i} ---> mfi_np  HEREEEE : {mfi_np}")
                mfi = mfi_np[-1]
                # print(f"mfi: {mfi}")
                # print(f"{type(mfi)} --> {mfi} +20: {mfi+20}")
                if mfi == np.nan:
                    windows = [[5, 20, 67]]
                    thresholds = [10]
                    sensitivity = [0.5]
                elif 80 > mfi:
                    windows = [[8,13, 21]]
                    thresholds = [5]
                    sensitivity = [0.95]
                elif mfi <= 20:
                    windows = [[5,8,13]]
                    thresholds = [7]
                    sensitivity = [0.9]
                else:
                    windows = [[5, 20, 67]]
                    thresholds = [10]
                    sensitivity = [0.5]
                # print(windows)
                return convert_to_array(windows), convert_to_type(thresholds, int), convert_to_type(sensitivity, float) 
            # CUSTOM PARAM FUNC ================================================

            dfmt = calc_tide_sig(dfm,
                                cols_set=[['open','high','low']],
                                param_func_tide= param_func_tide_MFI,
                                dynamic_param_col = ["MFI"])
            end = time.time()
            # print(f"------> Time taken : {np.round(end-start,3)} seconds\n")




            start = time.time()
            # print(f"\n{'='*30}\Z\n{'='*30}")
            # CUSTOM PARAM FUNC ================================================
            def param_func_Z_tide(x,i, col_index = 0):
                q_Lb, q_Ls, q_Sb, q_Ss = 0.9, 0.1, 0.1, 0.9

                # mfi = x[i+1-14:i+1,col_index[0]]
                tide = x[-1,col_index[0]]
                # print(mfi)
                if tide == 1:
                    windows = [24]
                    thresholds = [1.5]
                else:
                    windows = [288]
                    thresholds = [2]
                return convert_to_array(windows), convert_to_type(thresholds, int)
            # CUSTOM PARAM FUNC ================================================
            dfmtz = calc_z_sig(dfmt,
                            #  cols_set=[['open'],['high'],['low'],['close']],
                            cols_set=[["close"]],
                            param_func_Z = param_func_Z_tide,
                            #  param_func_Z = param_func_Z,
                            dynamic_param_col= ["tide"],
                            dynamic_param_combine = True,)
            end = time.time()
            # print(f"---> Time taken (z): {np.round(end-start,3)} seconds\n")

            dfmtz.index.name = "date_time"
            """
            # SIGNAL CALCULATION FOR 1 df (1 instrument, 1 timeframe) 
            """
            sig_dict[timeframe] = dfmtz # !!!!!!!!!!!!!!!!!!!! CHANGE THIS TO dfmtz
            # sig_dict[timeframe] = dfmtzTP # !!!!!!!!!!!!!!!!!!!! CHANGE THIS TO dfmtz
        sig_dicts[instrument] = sig_dict

    return sig_dicts