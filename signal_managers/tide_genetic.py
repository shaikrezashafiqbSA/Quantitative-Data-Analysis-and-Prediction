
from klines_managers import klines_ccxt
from signal_managers import indicators
from performance_analytics import metrics
from performance_analytics import backtest_plot
from backtesters import backtest


def make_tp_position_dict(lookback, qtl):
    tp_position_dict = {}
    for tp in ["TP1","TP2","TP3"]:
            tp_position_dict[tp] = {}
            for direction in ["long","short"]:
                    tp_position_dict[tp][direction] = {}
                    tp_position_dict[tp][direction]["lookback"] = lookback
                    tp_position_dict[tp][direction]["qtl"] = qtl
    return tp_position_dict



def init_tp_position_parameter_population():
        tp_position_dict = {}

        tp_position_list = []
        lookbacks = [3,6,9]
        qtls = [0.3, 0.6, 0.99]

        for lookback in lookbacks:
                for qtl in qtls:
                        tp_position_list.append(make_tp_position_dict(lookback, qtl))

        return tp_position_list

# =======================
# Load 1h klines
# =======================


def initialise_parent(df=None,
                      instruments =  ["ccxt_kucoin__BTC-USDT"],
                      timeframes = ["1h"],
                      instrument_to_trade = "ccxt_kucoin__BTC-USDT",
                      timeframe_to_trade = "1h",
                      signal_function = "default",
                      show_plots=False
                      ):
        # data preprocessing

        tp_position_dict = {"TP1": {"long":{"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]}, 
                                "short": {"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]},
                                },
                        "TP2": {"long":{"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]},
                                "short": {"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]},
                                },
                        "TP3": {"long":{"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]},
                                "short": {"lookbacks":[3,6,9], "qtls": [0.3, 0.6, 0.99]},
                        }
                }
        if df is None:

            tide_parameters_0 = {"sensitivity":10,
                            "threshold":10, 
                            "windows":[10,20,67], 
                            "suffix":"",
                            "TPs":tp_position_dict, 
                            "SL_penalty":1
                            }


            kline_manager = klines_ccxt.KlinesManagerCCXT()
            test = kline_manager.load_ohlcvs(instruments = instruments,
                                            timeframes = timeframes,
                                            since = "2020-01-01 00:00:00",
                                            limit = 1000, update=False)
            df = test[instrument_to_trade][timeframe_to_trade].copy()

            df = indicators.calc_tides(df, 
                                    sensitivity = tide_parameters_0["sensitivity"],
                                    thresholds = tide_parameters_0["threshold"],
                                    windows = tide_parameters_0["windows"],
                                    suffix="")
        else:
            df = df.copy()




        # ============================
        # Backtest parameters
        # ============================
        signals = [f"tide"]



        fee = 0.0001
        # fee = 0
        slippage = 0.0

        min_holding_period = 0
        max_holding_period = 1e6

        long_equity = 5e5
        long_notional=5e5
        short_equity = 5e5
        short_notional= 5e5

        signal_function="default"
        figsize = (20,15) #(20,29) # width, height
        # figsize = (20,30)
        # window =["2020-11-02","2022-08-23"] # - 0th backtest
        # window =["2010-01-01","2023-03-23"] 
        model_name= "Trend_Following"
        # window =["2022-06-01","2023-12-31"] 
        # model_name = "Mean_Reversion"


        tradable_times = [["00:05", "23:50"]]
        closing_session_times= [["23:50", "00:00"]]



        timeframe = "1h"
        sig_timeframe = f"1h"
        kline_to_trade = f"close"
        volume_to_trade= f"volume"



        title = f"{timeframe} BTC spot | fees: {fee*1e4} bps"
        file_name = f"{timeframe} BTC Tide"
        # signal = "ES_USD_close_sig" 
        # window = ["2021-01-01","2021-12-31"] 
        df_sig = df.copy()
        if signal_function == "default":
              df_sig["sig"] = df_sig[signals]
        df_trade = df_sig.copy()

        signals = df_trade.columns
        df_backtested,df_trades,df_summary = backtest.backtest(model_name= model_name,
                                                        df0=df_trade,
                                                        timeframe=timeframe,
                                                        kline_to_trade=kline_to_trade,
                                                        volume_to_trade = volume_to_trade,
                                                        tradable_times = tradable_times,
                                                        closing_session_times = closing_session_times,
                                                        position_sizing_to_trade=None,
                                                        min_holding_period=min_holding_period/int(timeframe[:-1]),
                                                        max_holding_period=max_holding_period/int(timeframe[:-1]),
                                                        fee=fee,
                                                        slippage=slippage,
                                                        long_equity = long_equity,
                                                        short_equity = short_equity,
                                                        long_notional=long_notional,
                                                        short_notional=short_notional,
                                                        signals=signals, 
                                                        signal_function=signal_function, 
                                                        figsize=figsize, # width, height
                                                        show_B=True,
                                                        show_LS=True,
                                                        title=title,
                                                        file_name=file_name,
                                                        plots=show_plots,
                                                        diagnostics_verbose=False,
                                                        trail_SL = None,
                                                        trail_TP = None,
                                                        trail_increment = None,
                                                        N=365*25
                                                        )

        df = df_backtested.copy()

        return df




import numpy as np
import pandas as pd
from signal_managers.indicators import calc_tide_strengths

def find_max_sharpe(df):
    max_sharpe = df['sharpe'].max()
    max_sharpe_row = df.loc[df['sharpe'] == max_sharpe]
    return max_sharpe_row

def evaluate(df0,s, t, w, tpc_val, sl, signal_function="strengths", window =["2020-01-01","2022-12-31"],
           title = f"Macro BTC spot | fees",
           file_name = f"Macro BTC Tide",
            plot=False ):
    df = df0.copy()
    df = indicators.calc_tides(df, 
                                sensitivity = s,
                                thresholds = t,
                                windows = w,
                                suffix="")
    
    df = calc_tide_strengths(df, penalty = sl, tp_position_dict = tpc_val)

    # =======================
    # BACKTEST
    # =======================
    signals = df.columns[-63:]
    fee = 0.0001
    # fee = 0
    slippage = 0.0

    min_holding_period = 0
    max_holding_period = 1e6

    long_equity = 5e5
    long_notional=5e5
    short_equity = 5e5
    short_notional= 5e5

#     signal_function="strengths"
    figsize = (20,15) #(20,29) # width, height
    # figsize = (20,30)
    # window =["2020-11-02","2022-08-23"] # - 0th backtest
    model_name= "Trend_Following"
    # window =["2022-06-01","2023-12-31"] 
    # model_name = "Mean_Reversion"


    tradable_times = [["00:05", "23:50"]]
    closing_session_times= [["23:50", "00:00"]]



    timeframe = "1h"
    sig_timeframe = f"1h"
    kline_to_trade = f"close"
    volume_to_trade= f"volume"



#     title = f"{timeframe} BTC spot | fees: {fee*1e4} bps"
#     file_name = f"{timeframe} BTC Tide"
    # signal = "ES_USD_close_sig" 
    # window = ["2021-01-01","2021-12-31"] 
    df_sig = df.copy()
    df_trade = df_sig[window[0]:window[1]].copy()

    # df_trade[["sig"]] = df_trade[signals].columns
    signals = df_trade.columns
    # signals = ["sig"]

    try: 
        df_backtested,df_trades,df_summary = backtest.backtest(model_name= model_name,
                                                        df0=df_trade,
                                                        timeframe=timeframe,
                                                        kline_to_trade=kline_to_trade,
                                                        volume_to_trade = volume_to_trade,
                                                        tradable_times = tradable_times,
                                                        closing_session_times = closing_session_times,
                                                        position_sizing_to_trade=None,
                                                        min_holding_period=min_holding_period/int(timeframe[:-1]),
                                                        max_holding_period=max_holding_period/int(timeframe[:-1]),
                                                        fee=fee,
                                                        slippage=slippage,
                                                        long_equity = long_equity,
                                                        short_equity = short_equity,
                                                        long_notional=long_notional,
                                                        short_notional=short_notional,
                                                        signals=signals, 
                                                        signal_function=signal_function, 
                                                        figsize=figsize, # width, height
                                                        show_B=True,
                                                        show_LS=True,
                                                        title=title,
                                                        file_name=file_name,
                                                        plots= plot,
                                                        diagnostics_verbose=False,
                                                        trail_SL = None,
                                                        trail_TP = None,
                                                        trail_increment = None,
                                                        N=365*8
                                                        )
    except Exception as e:
            df_backtested["tide"] = np.where((df_backtested["L_positions"]>0) & (df_backtested["tide"]>0),1,-1)
            df_backtested["tide"] = np.where((df_backtested["S_positions"]<0) & (df_backtested["tide"]<0),-1,1)
            df = df_backtested.copy()
            return df_backtested, df_summary

    sharpe = df_summary.iloc[0,0]

    # =======================
    # Mutate new tide strategy
    # =======================
    # has to have muateted something necessary for survival or remove unnecessary traits
    # 1. mutate the tide by adding a new window
    # 2. mutate the TP by adding a new window
    # 3. mutate the SL by adding a new window


    # Settings for long or short bias, since different trajectories with either
    df_backtested["tide"] = np.where((df_backtested["L_positions"]>0) & (df_backtested["tide"]>0),1,-1)
    df_backtested["tide"] = np.where((df_backtested["S_positions"]<0) & (df_backtested["tide"]<0),-1,1)



    df = df_backtested.copy()



    return df, df_summary


def evaluate_population(df, parameters, generations=6,signal_function="strengths", window =["2020-01-01","2022-12-31"], plot=True, verbose=True):
    sharpe_results = {}
    for i in range(generations):
        df, df_summary = evaluate(df,
                                  parameters['sensitivity'],
                                  parameters['threshold'],
                                  parameters['windows'],
                                  parameters['TPs'],
                                  parameters['SL_penalty'],
                                  signal_function=signal_function,
                                  window =window,
                                  plot = False,
                                )
        sharpe_results.update({i:{'sharpe':df_summary.iloc[0,0], 'df':df, 'df_summary':df_summary,'parameters':parameters}})
        if verbose: print(f"Sharpe: {df_summary.iloc[0,0]} | Sensitivity: {parameters['sensitivity']} | Thresholds: {parameters['threshold']} | Windows: {parameters['windows']} | TP: {parameters['TPs']} | SL: {parameters['SL_penalty']}")


    df = pd.DataFrame(sharpe_results).T
    if plot:
        df.plot(title=f"sensitivity: {parameters['sensitivity']}, window: {parameters['windows']}, threshold: {parameters['threshold']}", xlabel="generations", ylabel="sharpe ratio")
    max_sharpe_row = find_max_sharpe(df)
    best_df = max_sharpe_row['df'].values[0]
    best_df_summary = max_sharpe_row['df_summary'].values[0]
    best_parameters = max_sharpe_row['parameters'].values[0]
    return df, best_df,best_df_summary, best_parameters, max_sharpe_row
