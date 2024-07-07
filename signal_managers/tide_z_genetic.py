
from backtesters import backtest
from signal_managers.tpsl import calc_signal_tpsl

# def make_tp_position_dict(lookback, qtl):
#     tp_position_dict = {}
#     for tp in ["TP1","TP2","TP3"]:
#             tp_position_dict[tp] = {}
#             for direction in ["long","short"]:
#                     tp_position_dict[tp][direction] = {}
#                     tp_position_dict[tp][direction]["lookback"] = lookback
#                     tp_position_dict[tp][direction]["qtl"] = qtl
#     return tp_position_dict



# def init_tp_position_parameter_population():
#         tp_position_dict = {}

#         tp_position_list = []
#         lookbacks = [3,6,9]
#         qtls = [0.3, 0.6, 0.99]

#         for lookback in lookbacks:
#                 for qtl in qtls:
#                         tp_position_list.append(make_tp_position_dict(lookback, qtl))

#         return tp_position_list

# =======================
# Load 1h klines
# =======================

# import pandas as pd
# from klines_managers import klines_ccxt
# from signal_managers.resampler import calc_klines_resample

# from utils.list_type_converter import convert_to_type,convert_to_array
# from signal_managers.indicators import calc_mfi_sig, param_func_mfi_EMAVol, param_func_mfi, calc_signal_TPSL
# from signal_managers.indicators import calc_tide_sig, calc_tide_sig, param_func_tide_EMAVol, param_func_tide
# from signal_managers.indicators import calc_z_sig, param_func_Z_EMAVol, param_func_Z
# from signal_managers.composite_signals import merge_sig_dicts

# from signal_managers.mfi_tide_z_signals import load_klines
# from signal_managers.mfi_tide_z_signals import resample_instruments_dict
# from signal_managers.mfi_tide_z_signals import calc_mfi_tide_z_signals

# def initialise_parent(df=None,
#                       instruments =  [
#                                         "ccxt_kucoin__BTC-USDT",
#                                         "ccxt_kucoin__ETH-USDT",
#                                         "ccxt_currencycom__US500",
#                                         "ccxt_currencycom__DXY",
#                                         "ccxt_currencycom__Gold",
#                                         "ccxt_currencycom__Oil - Crude",
#                                         "ccxt_currencycom__NVDA",
#                                         "ccxt_currencycom__UVXY"
#                                         ],
#                       timeframes = ["1h"],
#                       timeframes_to_resample_to = ["2h","3h","4h","6h","8h","12h"],
#                       instrument_to_trade = "ccxt_kucoin__BTC-USDT",
#                       timeframe_to_trade = "1h",
#                       signal_function = "z_sig",
#                       signals_to_trade = "sig",
#                       backtest_window = ["2021-01-01","2023-12-31"],
#                       sig_lag=0,
#                       kline_to_trade = f"close",
#                       volume_to_trade= f"volume",
#                       fee = 0.001,
#                       slippage = 0.0,
#                       min_holding_period = 23,
#                       max_holding_period = 1e6,
#                       long_equity = 5e5,
#                       long_notional=5e5,
#                       short_equity = 5e5,
#                       short_notional= 5e5,
#                       show_plots=True, 
#                       figsize = (20,15), 
#                       chosen_to_debug = None
#                       ):

#     instruments_dict = load_klines(instruments = instruments, timeframes=timeframes)
        
         
#     instruments_dict = resample_instruments_dict(instruments_dict,
#                                              resample_to_list = timeframes_to_resample_to,
#                                              first_timeframe = timeframe_to_trade)

#     sig_dicts = calc_mfi_tide_z_signals(instruments_dict,chosen_to_debug=[])


#     df_to_trade = merge_sig_dicts(sig_dicts, 
#                         instrument_to_trade = instrument_to_trade,
#                         timeframe_to_trade = timeframe_to_trade,
#                         reduce_col_names = True)
    
#     df_backtested, df_trades, df_summary = objective_function(df_to_trade,
#                                                           backtest_window = ["2021-01-01","2023-12-31"],
#                                                           instrument_to_trade = "ccxt_kucoin__BTC-USDT",
#                                                           timeframe_to_trade = "1h",
#                                                           signal_function = "z_sig",
#                                                           signals_to_trade = "sig",
#                                                           sig_lag=0,
#                                                           kline_to_trade = f"close",
#                                                           volume_to_trade= f"volume",
#                                                           fee = 0.001,
#                                                           slippage = 0.0,
#                                                           min_holding_period = 23,
#                                                           max_holding_period = 1e6,
#                                                           long_equity = 5e5,
#                                                           long_notional=5e5,
#                                                           short_equity = 5e5,
#                                                           short_notional= 5e5,
#                                                           show_plots=True, 
#                                                           figsize = (20,15), 
#                                                           mutate_signals_w_TP = True,
#                                                           tradable_times = None, #[["00:05", "23:50"]]
#                                                           closing_session_times= None, #[["23:50", "00:00"]]
#                                                           )
    

    


# import numpy as np
# import pandas as pd
# from signal_managers.indicators import calc_tide_strengths

# def find_max_sharpe(df):
#     max_sharpe = df['sharpe'].max()
#     max_sharpe_row = df.loc[df['sharpe'] == max_sharpe]
#     return max_sharpe_row
import time
import numpy as np
import pandas as pd

def objective_function(df=None,
                        backtest_window = ["2021-01-01","2023-12-31"],
                        instrument_to_trade = "C:USDSGD",
                        timeframe_to_trade = "1m",
                        model_name= "Mean_Reversion",
                        signal_function = "z_sig",
                        signal_to_trade = "zscore",
                        sig_lags=[1,0],
                        kline_to_trade = f"close",
                        volume_to_trade= f"volume",
                        fees = [0.0002,0.00005],
                        slippages = [0.0,0.0],
                        min_holding_period = 0,
                        max_holding_period = 1e6,
                        min_holding_period_TP = 0,
                        max_holding_period_TP = 60,
                        position_sizing_to_trade = None,
                        long_equity= 10000,
                        long_notional= 10000,
                        short_equity= 10000,
                        short_notional= 10000,
                        show_plots = False, 
                        show_plots_TP = True,
                        figsize = (20,15), 
                        tradable_times =  [["00:00", "23:00"]],
                        days_of_the_week_to_trade = [0,1,2,3,4],
                        reduce_only = True,
                        run_signals_w_TP= True,
                        SL_penalty = 1,
                        tp_position_dict = {"TP1": {"L":{"lookback":30, "qtl": 0.35}, 
                                                        "S": {"lookback":30, "qtl":0.35}
                                                        },
                                                "TP2": {"L":{"lookback":30, "qtl": 0.65}, 
                                                        "S": {"lookback":30, "qtl":0.65}
                                                        },
                                                "TP3": {"L":{"lookback":30, "qtl": 0.95}, 
                                                        "S": {"lookback":30, "qtl":0.95}
                                                        }
                                                }, 
                        diagnostics_verbose   = False,       
                        show_rolling_sr = False,   
                        **kwargs):

        # =======================
        # BACKTEST
        # =======================
        # get N
        if diagnostics_verbose: print(f"\n{'='*30}\n(0) Get N")
        t0=time.time()
        Ndf = df.copy()
        Ndf["datetime"]=Ndf.index
        N = Ndf.groupby(Ndf["datetime"].dt.year).count().max().max()
        if diagnostics_verbose: print(f"N: {N} ---> {time.time()-t0}s")

        if diagnostics_verbose: print(f"\n{'='*30}\n(1) Generate base {signal_to_trade}-signal to backtest --> sig_lag: {sig_lags[0]}")
        t0 = time.time()

        title = f"{timeframe_to_trade} {instrument_to_trade} {signal_function} | fees: {fees[0]*1e4} bps"
        file_name = f"{timeframe_to_trade} {instrument_to_trade} Z"

        df_sig = df.copy()
        # df_sig["sig"] = df_sig[signals_to_trade]#.shift(-1).fillna(method="ffill")
        # df_trade = df_sig[backtest_window[0]:backtest_window[1]].copy()
        # signals_to_trade= ["sig",signals_to_trade]

        df_trade = df_sig[backtest_window[0]:backtest_window[1]].copy()
        # df_trade["sig"] = df_trade[signal_to_trade]#.shift(-1).fillna(method="ffill")
        signals_to_trade= list(df_trade.columns)

        df_backtested,df_trades,df_summary = backtest.backtest(model_name= model_name,
                                                                df0=df_trade,
                                                                timeframe=timeframe_to_trade,
                                                                kline_to_trade=kline_to_trade,
                                                                volume_to_trade = volume_to_trade,
                                                                tradable_times = tradable_times,
                                                                days_of_the_week_to_trade = days_of_the_week_to_trade,
                                                                position_sizing_to_trade=position_sizing_to_trade,
                                                                min_holding_period=min_holding_period,
                                                                max_holding_period=max_holding_period,
                                                                sig_lag=sig_lags[0],
                                                                fee=fees[0],
                                                                slippage=slippages[0],
                                                                long_equity = long_equity,
                                                                short_equity = short_equity,
                                                                long_notional=long_notional,
                                                                short_notional=short_notional,
                                                                signals=signals_to_trade, 
                                                                signal_function= signal_function, 
                                                                figsize=figsize, # width, height
                                                                show_B=True,
                                                                show_LS=True,
                                                                title=title,
                                                                file_name=file_name,
                                                                plots=show_plots,
                                                                diagnostics_verbose=diagnostics_verbose,
                                                                trail_SL = None,
                                                                trail_TP = None,
                                                                trail_increment = None,
                                                                N=N,
                                                                reduce_only=reduce_only,
                                                                # L_buy = -1, # THESE ARE INSTANTLY PASSED TO backtest.backtest
                                                                # L_sell = -1,
                                                                # S_buy = 1, 
                                                                # S_sell = 1,
                                                                **kwargs)
        t1 = time.time()
        t01 = np.round(t1-t0,2)
        if diagnostics_verbose: print(f"Time taken: {t01}s\n{'='*30}")

        if run_signals_w_TP: 
                if diagnostics_verbose: print(f"\n{'='*30}\n(2) TP based {signal_to_trade} signal")


                df_TP = calc_signal_tpsl(df_backtested,
                                        penalty = SL_penalty,
                                        tp_position_dict = tp_position_dict,)
                # df_sig["sig"] = df_sig[signals_to_trade]#.shift(-1).fillna(method="ffill")
                # print(dfmtzTP.filter(regex="sig_long_SL2")) there is sig_long_SL2 here 
                # df_trade["sig"] = df_trade[signals_to_trade]#.shift(-1).fillna(method="ffill")
                if diagnostics_verbose: print(f"Calculated TPs based off base backtest {time.time()-t1}s")

                signals_to_trade= list(df_TP.columns)

                title = f"{timeframe_to_trade} {instrument_to_trade} {signal_function}_TP | fees/slippage: {fees[1]*1e4} bps/ {slippages[1]*1e4} bps"
                file_name = f"{timeframe_to_trade} {instrument_to_trade} Z"
                df_backtested_TP, df_trades_TP, df_summary_TP = backtest.backtest(model_name= model_name,
                                                                                df0 = df_TP,
                                                                                timeframe=timeframe_to_trade,
                                                                                kline_to_trade = f"close",
                                                                                volume_to_trade= f"volume",
                                                                                tradable_times = tradable_times,
                                                                                days_of_the_week_to_trade = days_of_the_week_to_trade,
                                                                                position_sizing_to_trade = None,
                                                                                min_holding_period = min_holding_period_TP,
                                                                                max_holding_period = max_holding_period_TP, 
                                                                                sig_lag=sig_lags[1],
                                                                                fee = fees[1], 
                                                                                slippage = slippages[1],
                                                                                long_equity = long_equity,
                                                                                long_notional= long_notional,
                                                                                short_equity = short_equity,
                                                                                short_notional= short_notional,
                                                                                signals = signals_to_trade,
                                                                                signal_function = f"{signal_function}_TP",
                                                                                figsize = figsize,
                                                                                show_B = True,
                                                                                show_LS = True,
                                                                                title=title,
                                                                                file_name=file_name,
                                                                                plots = show_plots_TP,
                                                                                diagnostics_verbose=diagnostics_verbose,
                                                                                trail_SL = None,
                                                                                trail_TP = None,
                                                                                trail_increment = None,
                                                                                N=N,
                                                                                reduce_only=reduce_only,
                                                                                **kwargs
                                                                                )
                t2 = time.time()
                t12 = np.round(t2-t1,2)
                if diagnostics_verbose: print(f"Time taken: {t12}s\n{'='*30}")

                t02 = np.round(t2-t0,2)
                if diagnostics_verbose: print(f"TOTAL Time taken: {t02}s")
                return df_backtested_TP, df_trades_TP, df_summary_TP 

                
        else:
                return df_backtested, df_trades, df_summary

# def mutate_signals(df,sig_to_mutate, debug_verbose = True):
#     sig = sig_to_mutate

#     # START OUTPUT PRINTS ============================================================================================
#     if False:
#         cols_to_seek = ["L_id","sig","L_positions","L_entry_price","L_exit_price", "S_positions","S_entry_price","S_exit_price"]
#         L_id_to_seek = df["L_id"].iloc[20:40].min() #df["L_id"].dropna().iloc[-1]  
#         print(f"{'===='*20}\nBEFORE SIGNAL MUTATION\n{'===='*20}\n-->\n{df['L_id'].dropna().iloc[-1]}")
#         df_to_print = df[df["L_id"]==L_id_to_seek][cols_to_seek]
#         print(df_to_print.head(3))
#         print(df_to_print.tail(3))
#     # END PRINTS ======================================================================================================

#     # TP signal mutation
#     # L_TP_signal_concur = all(df["L_positions"].values>0)
#     # S_TP_signal_concur = all(df["S_positions"].values<0)
#     L_TP_signal_concur = np.where(df["L_positions"]>0,True,False)
#     S_TP_signal_concur = np.where(df["S_positions"]<0,True,False)
#     # print(df[sig].values>0)
#     # L_sig_concur = all(df[sig].values >0)
#     # S_sig_concur = all(df[sig].values >0)
#     L_sig_concur = np.where(df[sig]>2,True,False)
#     S_sig_concur = np.where(df[sig]<-2,True,False)
#     # print(f"L_TP_signal_concur: {L_TP_signal_concur[20:40]} ||| L_sig_concur: {L_sig_concur[20:40]}")
# #     print(f"L_TP_signal_concur: {np.shape(L_TP_signal_concur)} ||| L_sig_concur: {np.shape(L_sig_concur)} || len(df): {len(df)}")
# #     print(f"df: {np.shape(df)}")
#     # print(f"!!!!!!!!!!!!!!!!!!!!!! why is open 30? ---> {sig}")
#     df[sig] = np.where(S_TP_signal_concur & S_sig_concur,-2,0) + np.where(L_TP_signal_concur | L_sig_concur,2,0)
# #     df[sig[0]] = np.where(L_TP_signal_concur & L_sig_concur,2,-2)
#     # START OUTPUT PRINTS =============================================================================================
#     if False:
#         L_id_to_seek = df["L_id"].iloc[20:40].min()
#         print(f"{'===='*20}\nAFTER SIGNAL MUTATION\n{'===='*20}\n-->\n{L_id_to_seek}")
#         df_to_print = df[df["L_id"]==L_id_to_seek][cols_to_seek]
#         print(df_to_print.head(3))
#         print(df_to_print.tail(3))
#     # END PRINTS ======================================================================================================

#     return df

# def evaluate_population(df, parameters, generations=6,signal_function="strengths", window =["2020-01-01","2022-12-31"], plot=True, verbose=True):
#     sharpe_results = {}
#     for i in range(generations):
#         df, df_summary = evaluate(df,
#                                   parameters['sensitivity'],
#                                   parameters['threshold'],
#                                   parameters['windows'],
#                                   parameters['TPs'],
#                                   parameters['SL_penalty'],
#                                   signal_function=signal_function,
#                                   window =window,
#                                   plot = False,
#                                 )
#         sharpe_results.update({i:{'sharpe':df_summary.iloc[0,0], 'df':df, 'df_summary':df_summary,'parameters':parameters}})
#         if verbose: print(f"Sharpe: {df_summary.iloc[0,0]} | Sensitivity: {parameters['sensitivity']} | Thresholds: {parameters['threshold']} | Windows: {parameters['windows']} | TP: {parameters['TPs']} | SL: {parameters['SL_penalty']}")


#     df = pd.DataFrame(sharpe_results).T
#     if plot:
#         df.plot(title=f"sensitivity: {parameters['sensitivity']}, window: {parameters['windows']}, threshold: {parameters['threshold']}", xlabel="generations", ylabel="sharpe ratio")
#     max_sharpe_row = find_max_sharpe(df)
#     best_df = max_sharpe_row['df'].values[0]
#     best_df_summary = max_sharpe_row['df_summary'].values[0]
#     best_parameters = max_sharpe_row['parameters'].values[0]
#     return df, best_df,best_df_summary, best_parameters, max_sharpe_row
