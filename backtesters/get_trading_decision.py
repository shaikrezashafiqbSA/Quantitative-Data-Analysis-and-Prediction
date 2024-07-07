import os

from utils.pickle_helper import pickle_this

def run_backtest(data_params_payload, 
                 return_dfs= False, 
                 df = None, 
                 df_stage_name = None, #"df_stage_KGI_1m"
                 ):

    t_s = time.time()
    if df is None  and df_stage_name is None:
        df_stage_name = os.getenv('df_stage_name')
        df = pickle_this(pickle_name=df_stage_name, path="./optimization/staging/")
    elif (df is None) and df_stage_name is not None:
        df = pickle_this(pickle_name=df_stage_name, path="./optimization/staging/")



    dfmtz = calc_sig().calc_signal(
                                    df,
                                    data_params_payload,
                                )



    # print params

    df_backtested_TP, df_trades_TP, df_summary_TP = trading_decision().get_trading_decision(dfmtz,
                                                                                            data_params_payload = data_params_payload,
                                                                                            )

    config_dict = { "total": df_summary_TP.loc["Sharpe","total"],
                    "long": df_summary_TP.loc["Sharpe","longs only"],
                    "short":df_summary_TP.loc["Sharpe","shorts only"],
                    "data_params_payload": data_params_payload,}
    # print(f"={'='*50}\n{'=' + f'backtest run: {np.round(time.time()-t_s,3)}s'.center(48,' ') + '='}\n{'='*50}")
    if return_dfs:
        return config_dict, df_backtested_TP, df_trades_TP, df_summary_TP
    return config_dict





import traceback

from signal_managers.resampler import resample_instruments_dict

from klines_managers.klines_manager_agg import KlinesManagerAgg

class load_ohlcv:
    def __init__(self, 
                 window = ["2019-01-01","2026-12-31"],
                 instruments_to_query = ["KGI_USDSGD", "polygon__C:USDSGD"],
                 instruments=["USDSGD","C:USDSGD"],#, "C:EURUSD", "C:USDCAD", "C:USDJPY", "C:USDGBP"],
                 instrument_index_to_trade=0,  
                 timeframes=["1m"], 
                 resample_to_list=["30m"], #["5m", "30m", "1h", "4h", "1d"], 
                 since="2019-01-01 00:00:00", 
                 limits={"polygon": 5000, "ccxt": 5000}, 
                 update=False,
                 timeframe_to_trade="1m",
                 memory_len=None, 
                verbose=False,
                tradable_times = None,
                ):
        self.verbose = verbose
        self.window = window
        self.instruments = instruments
        self.timeframes = timeframes
        self.resample_to_list = resample_to_list
        self.since = since
        self.limits = limits
        self.update = update
        self.timeframe_to_trade = timeframe_to_trade
        self.instrument_index_to_trade = instrument_index_to_trade
        self.instrument_to_trade = self.instruments[self.instrument_index_to_trade]
        self.memory_len = memory_len
        self.instruments_to_query = instruments_to_query
        self.tradable_times = tradable_times

    def load_klines(self, 
                    return_dict = False, 
                    stage=False, 
                    stage_df_name = "df_stage", 
                    use_alt_volume=False, 
                    update = False,
                    clean_base_data_flag = False,
                    memory_len_lock = True,
                    return_raw_dict = False,
                    ):

        kline_manager_agg = KlinesManagerAgg(instruments = self.instruments_to_query,
                                                timeframes = self.timeframes,
                                                since = self.since,
                                                limits = self.limits,
                                                update=self.update)
        instruments_dict = kline_manager_agg.load_ohlcvs(update = update)

        if return_raw_dict:
            return instruments_dict
        # =======================================================================================================
        # Resample from 1hr --> 2h, 3h, 4h, 6h, 12h, 24h, 
        # =======================================================================================================
        instruments_dict = resample_instruments_dict(instruments_dict,
                                                    resample_to_list = self.resample_to_list, #["5m", "30m", "1h", "4h", "1d"],
                                                    first_timeframe = self.timeframes[0], 
                                                    clean_base_data_flag=clean_base_data_flag)

        if self.tradable_times is not None:
            for instrument in instruments_dict.keys():
                for timeframe in instruments_dict[instrument].keys():
                    instruments_dict[instrument][timeframe] = instruments_dict[instrument][timeframe].between_time(self.tradable_times[0][0], self.tradable_times[-1][-1])
        if return_dict:
            return instruments_dict


        # parameters
        # df = instruments_dict["C:USDSGD"]["1m"].copy()
        if self.verbose: print(f"BEFORE TRANSFORMATION: instrument_dict.keys() = {instruments_dict.keys()}")
        # rename all keys in instruments_dict to instruments

        # print(f"self.instruments = {self.instruments}")
        if memory_len_lock:
            if self.memory_len is None:
                df = instruments_dict[self.instrument_to_trade][self.timeframe_to_trade][self.window[0]:self.window[1]].copy()
            else:
                df = instruments_dict[self.instrument_to_trade][self.timeframe_to_trade].tail(self.memory_len).copy()
        else:
            df = instruments_dict[self.instrument_to_trade][self.timeframe_to_trade][self.window[0]:self.window[1]].copy()

        # USE 2nd loaded data's volume
        if use_alt_volume:
            df["volume"] = instruments_dict[self.instruments[1]][self.timeframe_to_trade][self.window[0]:self.window[1]]["volume"].copy()

        if stage:
            pickle_this(data=df, pickle_name=stage_df_name, path="./optimization/staging/")

        return df

import numba as nb
import numpy as np
import time

import importlib
from signal_managers import mfi 
importlib.reload(mfi)
from signal_managers.mfi import calc_signal_mfis

from signal_managers import tide
importlib.reload(tide)
from signal_managers.tide import calc_signal_tides
from signal_managers.tide import dynamic_params as tide_dynamic_params , static_params as tide_static_params

from signal_managers import zscore
importlib.reload(zscore)
from signal_managers.zscore import calc_signal_z
from signal_managers.zscore import dynamic_params as z_dynamic_params, static_params as z_static_params


class calc_sig(load_ohlcv):

    def __init__(self,):
        # get all attributes of test class
        super().__init__()

        # feature params
    def calc_signal(self, 
                    df,
                    params_payload,
                    verbose = False,
                    ):

        timeframe_to_trade=params_payload.get('timeframe_to_trade', "5m")
        MFI_window=params_payload.get('MFI_window', 60)
        MFI_max_lookback=params_payload.get('MFI_max_lookback', 80)
        MFI_sharpe_windows = params_payload.get('MFI_sharpe_windows', [8, 13, 21])
        MFI_sharpe_threshold = params_payload.get('MFI_sharpe_threshold', 5)
        MFI_sharpe_sensitivity = params_payload.get('MFI_sharpe_sensitivity', 0.5)
        MFI_sharpe_strong_level =  params_payload.get('MFI_sharpe_strong_level', 0.67)
        MFI_sharpe_weak_level = params_payload.get('MFI_sharpe_weak_level', 0.67)
        MFI_strong_windows = params_payload.get('MFI_strong_windows', [8, 13, 21])
        MFI_strong_threshold = params_payload.get('MFI_strong_threshold', 7)
        MFI_strong_sensitivity = params_payload.get('MFI_strong_sensitivity', 0.5)
        MFI_weak_windows = params_payload.get('MFI_weak_windows', [34,45,55])
        MFI_weak_threshold = params_payload.get('MFI_weak_threshold', 7)
        MFI_weak_sensitivity = params_payload.get('MFI_weak_sensitivity', 0.67)
        MFI_flat_windows = params_payload.get('MFI_flat_windows', [89,121,144])
        MFI_flat_threshold = params_payload.get('MFI_flat_threshold', 10)
        MFI_flat_sensitivity = params_payload.get('MFI_flat_sensitivity', 0.5)
        tide_strong_level = params_payload.get('tide_strong_level', 0.8)
        tide_weak_level = params_payload.get('tide_weak_level', 0.2)
        tide_strong_window = params_payload.get('tide_strong_window', 240)
        tide_strong_threshold = params_payload.get('tide_strong_threshold', 1.5)
        tide_weak_window = params_payload.get('tide_weak_window', 288)
        tide_weak_threshold = params_payload.get('tide_weak_threshold', 2)
        tide_flat_window =   params_payload.get('tide_flat_window', 288)
        tide_flat_threshold = params_payload.get('tide_flat_threshold', 2)
        z_dynamic_run = params_payload.get('z_dynamic_run', True)
        tide_dynamic_run= params_payload.get('tide_dynamic_run', True)
        tide_dynamic_alt = params_payload.get('tide_dynamic_alt', True)
        z_dynamic_alt = params_payload.get('z_dynamic_alt', True)

        timeframe_mult = int(timeframe_to_trade[:-1])
        self.MFI_window = MFI_window#/timeframe_mult
        self.MFI_max_lookback = MFI_max_lookback#/timeframe_mult

        # regen = True

        t0=time.time()

        # =============================================================================
        # 1) MFI
        # =============================================================================
        if verbose: print(f"{'='*50}\n{'=' + 'MFI'.center(48,' ') + '='}\n{'='*50}")
        # 60, 80
        dfm = calc_signal_mfis(df, window=self.MFI_window, label=self.MFI_window)

        # =============================================================================
        # 2) TIDES
        # =============================================================================
        if verbose: print(f"{'='*50}\n{'=' + 'TIDE'.center(48,' ') + '='}\n{'='*50}")

        """ 
        =======================================================================================================
        Dynamic Param Function
        =======================================================================================================
        """ 
        @nb.njit(cache=True)
        def dynamic_params(np_klines: np.array, column_indexes: np.array):

            np_windows = np.full((np_klines.shape[0], 3), np.nan)
            np_thresholds = np.full(np_klines.shape[0], np.nan)
            np_sensitivity = np.full(np_klines.shape[0], np.nan)
            max_lookback = MFI_max_lookback # 80

            for i in range(np_klines.shape[0]):
                # Calculate the volatility of the input data
                if i < max_lookback:
                    windows = MFI_sharpe_windows, # [8, 3, 21]
                    threshold = MFI_sharpe_threshold, # 5
                    sensitivity = MFI_sharpe_sensitivity # 0.5
                    np_windows[i] = np.array(windows) 
                    np_thresholds[i] = int(threshold)
                    np_sensitivity[i] = sensitivity
                    continue
                MFI = np_klines[i+1-max_lookback:i+1,column_indexes[-1]]
                MFI_std = np.std(MFI)
                if MFI_std == 0.0:
                    MFI_sharpe = 0
                else:
                    MFI_sharpe = np.mean(MFI)/MFI_std # This should be a EMA vol calculated outside\

                """
                        if 0.67 <= MFI_sharpe:
                            windows = [8,13,21]
                            threshold = 5
                            sensitivity = 0.5
                        elif MFI_sharpe < 0.67:
                            windows = [34,45,55]
                            threshold = 7
                            sensitivity = 0.5
                        else:
                            windows = [89,121,144]
                            threshold = 10
                            sensitivity = 0.5
                
                """
                if tide_dynamic_alt:
                    if MFI_sharpe > MFI_sharpe_strong_level:
                        windows = MFI_strong_windows
                        threshold = MFI_strong_threshold
                        sensitivity = MFI_strong_sensitivity
                    elif MFI_sharpe < MFI_sharpe_weak_level:
                        windows = MFI_weak_windows # [34,45,55]
                        threshold = MFI_weak_threshold # 7
                        sensitivity = MFI_weak_sensitivity #0.5
                    else:
                        windows = MFI_flat_windows # [89,121,144]
                        threshold = MFI_flat_threshold # 10
                        sensitivity = MFI_flat_sensitivity # 0.5
                else:
                    if MFI_sharpe <= MFI_sharpe_strong_level:
                        windows = MFI_strong_windows
                        threshold = MFI_strong_threshold
                        sensitivity = MFI_strong_sensitivity
                    elif MFI_sharpe < MFI_sharpe_weak_level:
                        windows = MFI_weak_windows # [34,45,55]
                        threshold = MFI_weak_threshold # 7
                        sensitivity = MFI_weak_sensitivity #0.5
                    else:
                        windows = MFI_flat_windows # [89,121,144]
                        threshold = MFI_flat_threshold # 10
                        sensitivity = MFI_flat_sensitivity # 0.5

                np_windows[i] = np.array(windows) # how to ensure all int? 20
                np_thresholds[i] = int(threshold) # 0.9
                np_sensitivity[i] = sensitivity # 0.1

            return np_windows,np_thresholds,np_sensitivity


        @nb.njit(cache=True)
        def tide_static_params(np_klines: np.array, column_indexes: np.array):
            windows = MFI_flat_windows # [8, 13, 21]
            threshold = MFI_flat_threshold #5
            sensitivity = MFI_flat_sensitivity #0.9
            np_windows = np.full((np_klines.shape[0], 3), np.nan)
            np_thresholds = np.full(np_klines.shape[0], np.nan)
            np_sensitivity = np.full(np_klines.shape[0], np.nan)
            for i in range(np_klines.shape[0]):
                np_windows[i] = np.array(windows) # how to ensure all int? 
                np_thresholds[i] = int(threshold)
                np_sensitivity[i] = sensitivity

            return np_windows,np_thresholds,np_sensitivity

        if tide_dynamic_run:
            calc_params = dynamic_params
        else:
            calc_params = tide_static_params

        dfmt = calc_signal_tides(dfm,
                                calc_params = tide_dynamic_params,
                                target_cols=['open','high','low'],
                                fixed_window = True,
                                suffix = "",
                                dynamic_param_col = [f"MFI_{self.MFI_window}"],
                                verbose = False)

        # =============================================================================
        # 3) ZSCORES
        # =============================================================================
        if verbose: print(f"{'='*50}\n{'=' + 'ZSCORE'.center(48,' ') + '='}\n{'='*50}")


        """
        =======================================================================================================
        Dynamic Param Function
        =======================================================================================================
        """
        @nb.njit(cache=True)
        def dynamic_params(np_klines: np.array, column_indexes: np.array):

            np_windows = np.full(np_klines.shape[0], np.nan)
            np_thresholds = np.full(np_klines.shape[0], np.nan)

            for i in range(np_klines.shape[0]):
                tide = np_klines[i,column_indexes[-1]]

                """
                        if (tide < 5) and (tide > -5):
                            window = 240
                            threshold = 1.5
                        else:
                            window = 240
                            threshold = 2
                """
                if z_dynamic_alt:
                    ## This yields 2.82B, 0.44TP
                    if (tide > tide_strong_level):
                        window = tide_strong_window # 240
                        threshold = tide_strong_threshold # 1.5
                    elif (tide < tide_weak_level):
                        window = tide_weak_window # 288
                        threshold =  tide_weak_threshold# 2
                    else:
                        window = tide_flat_window# 288
                        threshold = tide_flat_threshold # 2
                else:
                    if (tide < tide_strong_level) and (tide > tide_weak_level):
                        window = tide_weak_window # 240
                        threshold = tide_weak_threshold # 1.5
                    else:
                        window = tide_strong_window# 240
                        threshold = tide_strong_threshold # 2

                np_windows[i] = int(window)
                np_thresholds[i] = threshold

            return np_windows,np_thresholds

        @nb.njit(cache=True)
        def z_static_params(np_klines: np.array, column_indexes: np.array):
            windows = tide_flat_window 
            threshold = tide_flat_threshold
            np_windows = np.full(np_klines.shape[0], np.nan)
            np_thresholds = np.full(np_klines.shape[0], np.nan)
            for i in range(np_klines.shape[0]):
                np_windows[i] = int(windows) 
                np_thresholds[i] = int(threshold)

            return np_windows,np_thresholds
        
        if z_dynamic_run:
            calc_params = dynamic_params
        else:
            calc_params = z_static_params 

        dfmtz = calc_signal_z(dfmt,
                            calc_params = calc_params,
                                target_cols=['close'],
                                suffix = "",
                                dynamic_param_col = ["tide"],
                                verbose = False)


        # =============================================================================
        # i) print time
        # =============================================================================
        t1=time.time()
        if verbose: print(f"{'='*50}\n{'=' + 'Time'.center(48,' ') + '='}\n{'='*50}")
        if verbose: print(f"Time taken: {t1-t0:.2f} seconds")

        return dfmtz


# BASE WINNING
import importlib
from signal_managers import tide_z_genetic
from models import Mean_Reversion
from backtesters import backtest
from signal_managers import indicators
from performance_analytics import metrics, backtest_plot
from signal_managers import rolling_sr
importlib.reload(backtest)
importlib.reload(Mean_Reversion)
importlib.reload(tide_z_genetic)
importlib.reload(indicators)
importlib.reload(metrics)
importlib.reload(backtest_plot)
importlib.reload(rolling_sr)
from signal_managers import rolling_sr
from signal_managers.tide_z_genetic import objective_function

class trading_decision(calc_sig):
    def __init__(self,
                 ):
        # get all attributes of test class
        super().__init__()



    def get_trading_decision(self, dfmtz, data_params_payload):

        df_backtested_TP, df_trades_TP, df_summary_TP = objective_function(dfmtz,
                                                                            backtest_window = data_params_payload.get('backtest_window', ["2021-01-01","2023-01-01"]),
                                                                            instrument_to_trade = data_params_payload.get('instrument_to_trade', "USDSGD"),
                                                                            timeframe_to_trade = data_params_payload.get('timeframe_to_trade', "5m"),
                                                                            model_name= data_params_payload.get('model_name', "Mean_Reversion"),
                                                                            position_sizing_to_trade = data_params_payload.get('position_sizing_to_trade', None),
                                                                            signal_function = data_params_payload.get('signal_function', "z_sig"),
                                                                            signal_to_trade = data_params_payload.get('signal_to_trade', "zscore"),
                                                                            sig_lags = data_params_payload.get('sig_lags', [0,0]),
                                                                            kline_to_trade = data_params_payload.get('kline_to_trade', "5m"),
                                                                            volume_to_trade= data_params_payload.get('volume_to_trade', "volume"), # f"volume",
                                                                            fees = data_params_payload.get('fee',[0.000,0.000]), # OBJECTIVE 
                                                                            slippages = data_params_payload.get('slippage',[0.000,0.000]),
                                                                            min_holding_period = data_params_payload.get('min_holding_period',0), #23,
                                                                            max_holding_period = data_params_payload.get('max_holding_period',60), # 1e6,#26,
                                                                            long_equity = data_params_payload.get('long_equity', 10000),
                                                                            long_notional=data_params_payload.get('long_notional', 10000),
                                                                            short_equity = data_params_payload.get('short_equity', 10000),
                                                                            short_notional= data_params_payload.get('short_notional', 10000),
                                                                            show_plots = data_params_payload.get('show_plots', False),
                                                                            show_plots_TP = data_params_payload.get('show_plots_TP', False),
                                                                            figsize = (25,15), 
                                                                            run_signals_w_TP = data_params_payload.get('run_signals_w_TP', True),
                                                                            mutate_signals_w_TP = False,
                                                                            SL_penalty = data_params_payload.get('SL_penalty', 1),
                                                                            tp_position_dict = data_params_payload.get('tp_position_dict', None),
                                                                            tradable_times = data_params_payload.get('tradable_times', [["00:00","08:59"]]),
                                                                            days_of_the_week_to_trade = data_params_payload.get('days_of_the_week_to_trade', [0,1,2,3,4]),
                                                                            L_buy = data_params_payload.get('L_buy', -1),
                                                                            L_sell = data_params_payload.get('L_sell', -1),
                                                                            S_buy = data_params_payload.get('S_buy', 1),
                                                                            S_sell = data_params_payload.get('S_sell', 1),
                                                                            reduce_only=data_params_payload.get('reduce_only', False),
                                                                            show_rolling_SR = data_params_payload.get('show_rolling_SR', False),
                                                                            diagnostics_verbose = data_params_payload.get('diagnostics_verbose', False)
                                                                            )

        return df_backtested_TP, df_trades_TP, df_summary_TP