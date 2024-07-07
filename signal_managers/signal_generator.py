import math 
from decimal import Decimal, ROUND_DOWN

from utils.pickle_helper import pickle_this
from utils.get_time import get_dt_times_now
import time
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
import numba as nb
from logger.logger import setup_logger
logger = setup_logger(logger_file_name="trading_bot")
import importlib
from signal_managers import mfi 
importlib.reload(mfi)
from signal_managers.mfi import calc_signal_mfis

from signal_managers import tide
importlib.reload(tide)
from signal_managers.tide import calc_signal_tides
from signal_managers.tide import dynamic_params as tide_dynamic_params

from signal_managers import zscore
importlib.reload(zscore)
from signal_managers.zscore import calc_signal_z
from signal_managers.zscore import dynamic_params as z_dynamic_params

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

from backtesters.get_trading_decision import trading_decision
from order_managers.base_order_manager import base_order_manager as order_manager
from datetime import timedelta

class signal_generator:
    def __init__(self,
                    model_config,
                    ):
        
        self.model_config = model_config
        self.first_run = True

    def setup_model_config(self, model_config):
        """
        This sets up the model config
        model_name = "Mean_Reversion"
        account = "DTF"
        reporting_currency = "USD"
        order_type = "MARKET"
        instruments = ["C:USDSGD"]
        model_instruments = {}
        signals = {}
        equity_settings = {}

        {'running_signal': 2.0,
          'open': 1.35633, 
          'high': 1.35635, 
          'low': 1.3556, 
          'close': 1.35623, 
          'bid': None, 
          'ask': None, 
          'model': None, 
          'pair': None, 
          'time': None, 
          'time SGT': None, 
          'tradable_times': [['00:00', '09:00']], 
          'L': {'in_position': True, 
                'positions': 0, 
                'side': None, 
                'model_id': nan, 
                'entry_now': False, 
                'entry_pending': False, 
                'entry_price': nan, 
                'entry_price_max': None, 
                'entry_price_filled': nan, 
                'entry_quantity': nan, 
                'entry_quantity_filled': nan, 
                'entry_time': Timestamp('2023-11-08 03:46:00'), 
                'exit_now': False, 
                'exit_pending': False, 
                'exit_price': None, 
                'exit_price_min': None, 
                'exit_price_filled': None, 
                'exit_quantity': None, 
                'exit_quantity_filled': None, 
                'exit_time': None, 
                'TP1': 1.354648, 'TP2': 1.354723, 'SL1': nan, 'SL2': nan, 
                'running_TP1': 1.355946, 'running_TP2': 1.356037, 'running_SL1': nan, 'running_SL2': nan, 
                'model': 'Mean_Reversion', 
                'pair': 'C:USDSGD', 
                'time': '2023-11-08 09:30:03.918 UTC', 
                'time SGT': '2023-11-08 17:30:03.918 SGT', 
                'running_signal': 2.0
                }, 
            'S': {'in_position': False, 
                    'positions': 0, 
                    'side': 'exit', 
                    'model_id': 33.0, 
                    'entry_now': False, 
                    'entry_pending': False, 
                    'entry_price': 1.35523, 
                    'entry_price_min': 1.355162, 
                    'entry_price_filled': 1.35523, 
                    'entry_quantity': 10000.0, 
                    'entry_quantity_filled': 10000.0, 
                    'entry_time': '2023-11-08 08:56:03.559 UTC', 
                    'exit_now': False, 
                    'exit_pending': False, 
                    'exit_price': 1.35559, 
                    'exit_price_max': 1.355658, 
                    'exit_price_filled': 1.35549, 
                    'exit_quantity': 10002.656376, 
                    'exit_quantity_filled': None, 
                    'exit_time': '2023-11-08 09:02:03.765 UTC', 
                    'TP1': 1.355335, 'TP2': 1.355153, 'SL1': 1.355698, 'SL2': 1.355775, 
                    'running_TP1': 1.355335, 'running_TP2': 1.355153, 'running_SL1': 1.355698, 'running_SL2': 1.355775, 
                    'model': 'Mean_Reversion', 
                    'pair': 'C:USDSGD', 
                    'time': '2023-11-08 09:30:03.944 UTC', 
                    'time SGT': '2023-11-08 17:30:03.944 SGT', 
                    'running_signal': 2.0
                    }
                }

        """
        for key, value in model_config.items():
            if isinstance(value, dict):
                print(f"key: {key} is a dict -->{value}")
                nested_dict = getattr(self, key, {})
                for nested_key, nested_value in value.items():
                    nested_dict[nested_key] = nested_value
                setattr(self, key, nested_dict)
            else:
                setattr(self, key, value)        

    def initialise(self, 
                   klines_manager,
                   model_state, 
                   debug=False, 
                   rewind_to = None): # ["2023-01-01","2023-02-01"]):
        # TODO: HAVE TO CLARIFY WHAT IS HAPPENING HERE - 4 scenarios
        # Better way is switch

        if rewind_to is None:
            df = self.load_klines(klines_manager, rewind_to=None)
        else:
            df, df_to_rewind = self.load_klines(klines_manager, rewind_to=rewind_to)

        self.dfmtz = self.calc_signal(df)
        if debug:
            model_state, df_backtested_TP, df_trades_TP, df_summary_TP = self.get_trading_decision(self.dfmtz, model_state, debug=True)
            return model_state, df_backtested_TP, df_trades_TP, df_summary_TP
        elif rewind_to is not None:
            model_state, df_backtested_TP, df_trades_TP, df_summary_TP = self.get_trading_decision(self.dfmtz, model_state, debug=False)
            return model_state, self.dfmtz, df_to_rewind
        else:
            model_state, df_backtested_TP, df_trades_TP, df_summary_TP = self.get_trading_decision(self.dfmtz, model_state)
            return model_state, df_backtested_TP, df_trades_TP, df_summary_TP

    def load_klines(self,
                    klines_manager,
                    rewind_to=None,  # ["2023-11-05", "2023-11-07"]
                    ):
        instruments_dict = klines_manager.load_ohlcvs(instruments=self.model_config["instruments"],
                                                        timeframes=self.model_config["timeframes"],
                                                        since=self.model_config["load_data_from"],
                                                        limit=5000,
                                                        update=True)

        # df = instruments_dict["C:USDSGD"]["1m"].copy()
        instrument_to_trade_str = self.model_config["instruments"][self.model_config["instrument_index_to_trade"]]
        timeframe_to_trade_str = self.model_config["timeframe_to_trade"]
        if rewind_to is None:
            self.df = instruments_dict[instrument_to_trade_str][timeframe_to_trade_str].tail(self.model_config["memory_len"]).copy()
            return self.df
        else:
            rewind_start_buffer = str(pd.to_datetime(rewind_to[0]) - timedelta(minutes=self.model_config["memory_len"]))
            rewind_end_buffer = rewind_to[0]
            self.df = instruments_dict[instrument_to_trade_str][timeframe_to_trade_str].loc[
                      rewind_start_buffer:rewind_end_buffer].copy()
            df_to_rewind = instruments_dict[instrument_to_trade_str][timeframe_to_trade_str].loc[self.df.index[-1]:].iloc[1:-1,:].copy()

            logger.info(f"\n{'='*90}\n{'=' + 'REWINDING'.center(48,' ') + '='}\n{'='*90}")
            logger.info(f"\n{self.df.index[0]} --> {self.df.index[-1]}")
            logger.info(f"\n{df_to_rewind.index[0]} --> {df_to_rewind.index[-1]}")
            logger.info(f"\n{'='*90}\n{'=' + 'REWINDING'.center(48,' ') + '='}\n{'='*90}")
            return self.df, df_to_rewind





    def calc_signal(self, df):
        # Params to be extracted out to model_config
        MFI_window = 60

        """
        # =============================================================================
        # 1) MFI
        # =============================================================================
        """
        logger.info(f"\n{'='*50}\n{'=' + 'SIGNALS SUITE'.center(48,' ') + '='}\n{'='*50}")
        logger.info(f"\n{'='*50}\n{'=' + 'MFI'.center(48,' ') + '='}\n{'='*50}")

        dfm = calc_signal_mfis(df, window=MFI_window, label=MFI_window)

        """
        # =============================================================================
        # 2) TIDES
        # =============================================================================
        """
        logger.info(f"\n{'='*50}\n{'=' + 'TIDE'.center(48,' ') + '='}\n{'='*50}")


        dfmt = calc_signal_tides(dfm,
                                calc_params = tide_dynamic_params,
                                target_cols=['open','high','low'],
                                fixed_window = True,
                                suffix = "",
                                dynamic_param_col = [f"MFI_{MFI_window}"],
                                verbose = True)
        """                       
        # =============================================================================
        # 3) ZSCORES DAN STRAT
        # =============================================================================
        """
        logger.info(f"\n{'='*50}\n{'=' + 'ZSCORE'.center(48,' ') + '='}\n{'='*50}")

        dfmtz = calc_signal_z(dfmt,
                            calc_params = z_dynamic_params,
                                target_cols=['close'],
                                suffix = "",
                                dynamic_param_col = ["tide"],
                                verbose = True)
        logger.info(f"\n{'='*50}\n{'=' + 'SIGNALS SUITE COMPLETE'.center(48,' ') + '='}\n{'='*50}")
        return dfmtz
    
        
    
    def update(self, msg, model_state, order_manager):
        """
        OHLCV bars data channel: CA
        msg = {'ev': 'CA', 'pair': 'USD/SGD', 'v': 78, 'o': 1.3701, 'c': 1.37012, 'h': 1.37014, 'l': 1.3696, 's': 1698824700000, 'e': 1698824760000}
        """
        # where 
        # now we have to extract open, high, low, close, volume, vwap (optional, if none then nan), close_time
        # and append this to df from self.load_klines()
        # then we have to run self.calc_signal(df)
        # then we have to run self.get_trading_decision(dfmtz)
        dt0 , dt0_SGT = get_dt_times_now()
        logger.info(f"\n{'='*100}\n{'=' + f'(B) Update ({dt0})'.center(98,' ') + '='}\n{'='*100}")
        logger.info(f"\n\nmodel_state before backtest:\n\n{model_state}\n")

        model_state = order_manager.update_model_state(model_state, order_type = self.order_type)

        logger.info(f"\n{'='*100}\n{'=' + f'(B) Update ({dt0})'.center(98,' ') + '='}\n{'='*100}")
        logger.info(f"\n\nmodel_state before backtest:\n\n{model_state}\n")
        model_state = self.round_dict(model_state, self.model_config["tick_size"], self.model_config["precision"])

        appending_data = {"datetime":pd.to_datetime(msg["s"], unit="ms"),
                            "open": msg["o"],
                            "high": msg["h"],
                            "low": msg["l"],
                            "close": msg["c"],
                            "volume": msg["v"],
                            "vwap": None,
                            "close_time": msg["e"],
                            "timestamp":msg["e"]/1000}

        appending_df = pd.DataFrame(appending_data, index=[0])
        logger.info(f"\n{'='*100}\nappending_df:\n{appending_df}\n{'='*100}\n")
        self.dfmtz.loc[appending_data["datetime"],:] = appending_df.loc[0,:]

        # TRIM 
        self.dfmtz = self.dfmtz.tail(self.memory_len).copy()

        logger.info(self.dfmtz.tail(10))
        logger.info(f"\n{'='*100}\n{'=' + f'(B) Update'.center(98,' ') + '='}\n{'='*100}")

        return {"dfmtz":self.dfmtz, "model_state":model_state}
    


    def get_trading_decision(self, dfmtz, model_state, model_params, debug=False, verbose=True, price_range =  0.00005):
        dt0 , dt0_SGT = get_dt_times_now()
        # if verbose: logger.info(f"\n{'='*100}\n{'=' + f'(A) Trading Decision ({dt0})'.center(98,' ') + '='}\n{'='*100}")
        # if verbose: logger.info(f"\n\nBEFORE FORWARD TEST: \n{dfmtz.tail(2)}\n\n")

        df_backtested_TP, df_trades_TP, df_summary_TP = trading_decision().get_trading_decision(dfmtz,
                                                                                                data_params_payload = model_params,
                                                                                                )
        
        
        # logger.info(f"\n---> df_summary_TP\n")
        # logger.info(f"\n     Total Sharpe - {df_summary_TP.loc['Sharpe','total']}")
        # logger.info(f"\n     Long Sharpe - {df_summary_TP.loc['Sharpe','longs only']}")
        # logger.info(f"\n     Short Sharpe - {df_summary_TP.loc['Sharpe','shorts only']}")
        TP_metrics = f"A: {df_summary_TP.loc['Sharpe','total']} || L: {df_summary_TP.loc['Sharpe','longs only']} || S: {df_summary_TP.loc['Sharpe','shorts only']}"
        # logger.info(f"\n{'='*100}\n{'=' + '(B) Trading Decision'.center(98,' ') + '='}\n{'=' + f'{TP_metrics}'.center(98,' ') + '='}\n{'='*100}")
        
        # logger.info(f"{model_state}\n\n")

        """
        # Check if theres a new trade in df_trades_TP
        # TODO: GOTTA REFORMAT THIS TO ONLY LONGS OR SHORTS WITH 
        # - IDS
        # - POSITION
        # - trade PRICE (range) 
        # - TP 
        # - SL
        # - SIG 
        """
        if debug:
            return df_trades_TP
        
        # ======================================================================================================
        # UPDATE MODEL STATE WITH NEW TRADES
        # ======================================================================================================
        # price_range =  
        if self.first_run:
            self.first_run = False
            model_state["tradable_times"] = self.model_config["tradable_times"]
            print(f"\n{'='*100}\n{'=' + 'FIRST RUN'.center(98,' ') + '='}\n{'='*100}\n{model_state['tradable_times']}")   
            for position in ["L", "S"]:
                time_actual, time_actual_SGT = get_dt_times_now()
                time_i = df_backtested_TP.index[-1].tz_localize('UTC')
                time_i_SGT = time_i.astimezone(pytz.timezone("Singapore")).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
                dt0_timestamp = datetime.strptime(dt0, '%Y-%m-%d %H:%M:%S.%f %Z')
                model_state[position].update({
                    "model": self.model_config['model_name'],
                    "pair": self.model_config["instrument_to_trade"],
                    "time": str(time_i),
                    "time SGT": str(time_i_SGT),
                    "time_actual": time_actual,
                    "time_actual SGT": time_actual_SGT,
                    # "in_position": False,
                    "positions": 0,
                    "entry_price_max": None,
                    "entry_price_min": None,
                    "entry_now": False,
                    "exit_now": False,
                })

                # If no trades then how?! ans: 
                latest_trades = pd.merge(df_trades_TP, df_backtested_TP[["zscore", "open", "high", "low", "close",]], left_index=True, right_index=True, how="left").tail(4)
                print(f"\n\ndf_trades_TP: \n{df_trades_TP}\n\nlatest_trades: \n{latest_trades}\n\n")
                model_state.update({
                    "open": df_backtested_TP["open"].iloc[-1],
                    "high": df_backtested_TP["high"].iloc[-1],
                    "low": df_backtested_TP["low"].iloc[-1],
                    "close": df_backtested_TP["close"].iloc[-1],
                    "running_signal": df_backtested_TP["zscore"].iloc[-1],
                })

                if len(latest_trades) >0:
                    last_trade_id = latest_trades[f"{position}_id"].iloc[-1]
                    # 
                    # model_state[position]["in_position"] = False if latest_trades[f"{position}_positions"].fillna(0).iloc[-1] == 0 else True
                    model_state[position]["positions"] = latest_trades[f"{position}_positions"].iloc[-1]
                    if model_state[position]["in_position"] or model_state[position]["positions"] != 0:
                        model_state[position].update({
                            "model_id": last_trade_id,
                            "entry_price_expected": latest_trades[f"{position}_entry_price"].iloc[-1],
                            "entry_quantity_expected": latest_trades[f"{position}_cost"].iloc[-1],
                            f"entry_time": latest_trades[f"{position}_entry_price"].index[-1],
                        })
                    if model_state[position]["positions"] == 1:
                        model_state[position]["entry_price_max"] = np.round(latest_trades["L_entry_price"].iloc[-1] * (1 + price_range), self.model_config["precision"])
                    elif model_state[position]["positions"] == -1:
                        model_state[position]["entry_price_min"] = np.round(latest_trades["S_entry_price"].iloc[-1] * (1 - price_range), self.model_config["precision"])
                    time_since_last_trade = (dt0_timestamp - latest_trades.index[-1]).total_seconds()
                    if time_since_last_trade > 0:
                        model_state[position]["entry_now"] = False
                        model_state[position]["exit_now"] = False
            return model_state, df_backtested_TP, df_trades_TP, df_summary_TP
        

        if not self.first_run and len(df_trades_TP) > 0:
            model_state["tradable_times"] = self.model_config["tradable_times"]
            for position in ["L", "S"]:
                time_actual, time_actual_SGT = get_dt_times_now()
                time_i = df_backtested_TP.index[-1].tz_localize('UTC')
                time_i_SGT = time_i.astimezone(pytz.timezone("Singapore")).strftime("%Y-%m-%d %H:%M:%S.%f %Z")

                model_state[position].update({
                    "model": self.model_config['model_name'],
                    "pair": self.model_config['instrument_to_trade'],
                    "time": str(time_i),
                    "time SGT": str(time_i_SGT),
                    "time_actual": time_actual,
                    "time_actual SGT": time_actual_SGT,
                })
                latest_trades = pd.merge(df_trades_TP.filter(regex=f"{position}_"), df_backtested_TP[["zscore", "open", "high", "low", "close",]], left_index=True, right_index=True, how="left").tail(2)
                if verbose:
                    logger.info(f"\n{'-'*100}\nLATEST TRADES:\n{latest_trades}\n{'-'*100}\n")
                model_state.update({
                    "open": latest_trades["open"].iloc[-1],
                    "high": latest_trades["high"].iloc[-1],
                    "low": latest_trades["low"].iloc[-1],
                    "close": latest_trades["close"].iloc[-1],
                    "running_signal": latest_trades[f"zscore"].iloc[-1],
                })

                
                # L_entry_condition = not model_state[position]["in_position"] and model_state[position]["positions"] == 0 and latest_trades[f"{position}_positions"].iloc[-1] > 0
                # L_exit_condition = model_state[position]["in_position"] and model_state[position]["positions"] == 1 and latest_trades[f"{position}_positions"].iloc[-1] == 0
                # S_entry_condition = not model_state[position]["in_position"] and model_state[position]["positions"] == 0 and latest_trades[f"{position}_positions"].iloc[-1] < 0
                # S_exit_condition = model_state[position]["in_position"] and model_state[position]["positions"] == -1 and latest_trades[f"{position}_positions"].iloc[-1] == 0
                triggerable_model_state_positions = (model_state[position]["positions"] == 0) or  math.isnan(model_state[position]["positions"])
                L_entry_condition = triggerable_model_state_positions and latest_trades[f"{position}_positions"].iloc[-1] > 0
                L_exit_condition =  model_state[position]["in_position"] and model_state[position]["positions"] == 1 and latest_trades[f"{position}_positions"].iloc[-1] == 0
                L_exit_condition_model_only = (not model_state[position]["in_position"]) and model_state[position]["positions"] == 1 and latest_trades[f"{position}_positions"].iloc[-1] == 0

                S_entry_condition = triggerable_model_state_positions and latest_trades[f"{position}_positions"].iloc[-1] < 0
                S_exit_condition =  model_state[position]["in_position"] and model_state[position]["positions"] == -1 and latest_trades[f"{position}_positions"].iloc[-1] == 0
                S_exit_condition_model_only = (not model_state[position]["in_position"]) and model_state[position]["positions"] == -1 and latest_trades[f"{position}_positions"].iloc[-1] == 0
            
                if position == "L":
                    if L_entry_condition:
                        model_state["L"].update({
                            "entry_now": True,
                            "positions": 1,
                            "side": "entry",
                            "model_id": latest_trades["L_id"].iloc[-1],
                            "entry_price_expected": latest_trades[f"{position}_entry_price"].iloc[-1],
                            "entry_quantity_expected": latest_trades[f"{position}_cost"].iloc[-1],
                        })
                        model_state["L"].update({"entry_price_max": np.round(model_state["L"]["entry_price_expected"]*(1+price_range), self.model_config["precision"]),})
                    elif L_exit_condition or L_exit_condition_model_only:
                        model_state["L"].update({
                            "exit_now": True if L_exit_condition else False,
                            "positions": 0, #if L_exit_condition else np.nan,
                            "side": "exit",
                            "model_id": latest_trades["L_id"].iloc[-1],
                            "exit_price_expected": latest_trades[f"{position}_exit_price"].iloc[-1],
                            "exit_quantity_expected": latest_trades[f"{position}_cost"].iloc[-1],
                        })
                        model_state["L"].update({"exit_price_min": np.round(latest_trades['L_exit_price'].iloc[-1]*(1-price_range), self.model_config["precision"]),})
                elif position == "S":
                    if S_entry_condition:
                        model_state["S"].update({
                            "entry_now": True,
                            "positions": -1,
                            "side": "entry",
                            "model_id": latest_trades["S_id"].iloc[-1],
                            "entry_price_expected": latest_trades[f"{position}_entry_price"].iloc[-1],
                            "entry_quantity_expected": latest_trades[f"{position}_cost"].iloc[-1],
                        })
                        model_state["S"].update({"entry_price_min": np.round(model_state["S"]["entry_price_expected"]*(1-price_range), self.model_config["precision"]),})
 
                    elif S_exit_condition or S_exit_condition_model_only:
                        model_state["S"].update({
                            "exit_now": True if S_exit_condition else False,
                            "positions": 0,# if S_exit_condition else np.nan,
                            "side": "exit",
                            "model_id": latest_trades["S_id"].iloc[-1],
                            "exit_price_expected": latest_trades[f"{position}_exit_price"].iloc[-1],
                            "exit_quantity_expected": latest_trades[f"{position}_cost"].iloc[-1],
                        })
                        model_state["S"].update({"exit_price_max": np.round(latest_trades['S_exit_price'].iloc[-1]*(1+price_range), self.model_config["precision"]),})
                    # else:

                model_state = self.round_dict(model_state, self.model_config["tick_size"], self.model_config["precision"])
            return model_state, df_backtested_TP, df_trades_TP, df_summary_TP
        else:
            model_state = self.round_dict(model_state, self.model_config["tick_size"], self.model_config["precision"])
            return model_state, df_backtested_TP, df_trades_TP, df_summary_TP


    def round_to_tick(self, price, tick_size, precision):
        return round(price, precision)


    def round_dict(self, d, tick_size, precision, keys_to_skip = ["model_id","running_signal"]):
        for k, v in d.items():
            if k in keys_to_skip:
                continue
            elif isinstance(v, dict):
                self.round_dict(v, tick_size, precision)
            elif isinstance(v, float) and not math.isnan(v):
                d[k] = self.round_to_tick(v, tick_size, precision)
        return d
