import pandas as pd
import datetime
import time


from klines_managers import klines_polygon
from signal_managers.mfi_tide_z_signals import resample_instruments_dict

from signal_managers.mfi import calc_signal_mfis
from signal_managers.tide import calc_signal_tides
from signal_managers.tide import dynamic_params, static_params
from signal_managers.zscore import calc_signal_z
from signal_managers.zscore import dynamic_params, static_params

from signal_managers.tide_z_genetic import objective_function
from slack_managers import messenger

class MRSignal:
    def __init__(self, 
                 instruments=["C:USDSGD"],
                 timeframes = ["1m"],
                 since = "2019-01-01 00:00:00"
                 ) -> None:
        self.instruments = instruments
        self.timeframes = timeframes
        self.since = since
        self.df = None

        # 
        self.load_klines()

        
    def load_klines(self):
        self.kline_manager_polygon = klines_polygon.KlinesManagerPolygon()
        self.instruments_dict = self.kline_manager_polygon.load_ohlcvs(instruments = self.instruments,
                                                                timeframes = self.timeframes,
                                                                since = self.since,
                                                                limit = 5000, update=True)
        self.df = self.instruments_dict["C:USDSGD"]["1m"].copy()
        
    def update_klines(self, msg):
        # msg = [{"ev":"CA","pair":"USD/SGD","v":72,"o":1.36503,"c":1.365,"h":1.36505,"l":1.3645,"s":1698116400000,"e":1698116460000}]
        new_klines = msg[0]
        s_datetime = pd.to_datetime(new_klines["s"], unit="ms")
        e_datetime = pd.to_datetime(new_klines["e"], unit="ms")

        new_klines_dict = {"datetime":s_datetime, 
                        "open":new_klines["o"],
                        "high":new_klines["h"],
                        "low":new_klines["l"],
                        "close":new_klines["c"],
                        "volume":new_klines["v"]}
        
        self.df.loc[new_klines_dict['datetime']] = [new_klines_dict['open'],
                                                    new_klines_dict['high'], 
                                                    new_klines_dict['low'],
                                                    new_klines_dict['close'],
                                                    new_klines_dict['volume'],
                                                    None,
                                                    new_klines["e"]
                                                    ]
        
        
    
    def load_notifier(self):
            messenger.send_msg("USDSGD_MR_Bot firing up ...")


    def run_signal_suite(self):
        instruments_dict = resample_instruments_dict(instruments_dict,
                                            resample_to_list = [],
                                            first_timeframe = "1m")
        MFI_window = 60
        timeframe_to_trade = "1m"
        instrument_to_trade = "C:USDSGD"
        signal_to_trade = "zscore"

        now = datetime.datetime.now()
        t0 = time.time()
        print(f"{now} --- Running signal suite for {instrument_to_trade} on {timeframe_to_trade} for {signal_to_trade}...")

        df = instruments_dict["C:USDSGD"][timeframe_to_trade].copy()

        dfm = calc_signal_mfis(df, window=MFI_window, label=MFI_window)

        dfmt = calc_signal_tides(dfm,
                    calc_params = dynamic_params,
                    target_cols=['open','high','low'],
                    fixed_window = True,
                    suffix = "",
                    dynamic_param_col = [f"MFI_{MFI_window}"],
                    verbose = True)
        
        dfmtz = calc_signal_z(dfmt,
                    #  calc_params = static_params,
                    calc_params = dynamic_params,
                    target_cols=['close'],
                    suffix = "",
                    dynamic_param_col = ["tide"],
                    verbose = True)
        
        df_backtested0, df_trades0, df_summary0 = objective_function(dfmtz,
                                                    backtest_window = ["2023-01-01","2023-12-31"],
                                                    instrument_to_trade = instrument_to_trade,
                                                    timeframe_to_trade = timeframe_to_trade,
                                                    signal_function = "z_sig",
                                                    signal_to_trade = signal_to_trade,
                                                    sig_lag=0,
                                                    kline_to_trade = f"close",
                                                    volume_to_trade= f"volume",
                                                    fee = 0.0001, # benchmark to beat is 0.001 2 ways so ours should be 0.0005 here
                                                    slippage = 0.0,#0.0001,
                                                    min_holding_period = 0, #23,
                                                    max_holding_period = 1e6, # 1e6,#26,
                                                    long_equity = 5e5,
                                                    long_notional=5e5,
                                                    short_equity = 5e5,
                                                    short_notional= 5e5,
                                                    show_plots=True, 
                                                    figsize = (20,15), 
                                                    run_signals_w_TP=False,
                                                    mutate_signals_w_TP = False,
                                                    tradable_times = [["00:00", "23:00"]],
                                                    days_of_the_week_to_trade = [0,1,2,3,4],
                                                    L_buy = -1,
                                                    L_sell = -1,
                                                    S_buy = 1, 
                                                    S_sell = 1,
                                                    )