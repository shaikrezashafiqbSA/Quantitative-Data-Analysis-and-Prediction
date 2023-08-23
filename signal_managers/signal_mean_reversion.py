from logger.logger import logger

from utils import pickle_helper
from signals import signals,composite_signals, signal_thresholds

class Signal_Mean_Reversion:
    def __init__(self,
                 model_config: dict):
        
        self.update_model_config(model_config)
    
    
        
    def update_model_config(self, model_config):
        self.runtime_window = model_config["signals"]["runtime_window"]
        self.target_signal = model_config["signals"]["target_signal"]
        
        # model instruments
        self.timeframes = model_config["model_instruments"]["timeframes"]
        self.smallest_timeframe = self.timeframes[0]
        self.spot_instrument = model_config["model_instruments"]["spot_instrument"]
        self.futures_instruments = model_config["model_instruments"]["futures_instruments"]
        self.factor_instruments = model_config["model_instruments"]["factor_instruments"]
        self.instruments = model_config["model_instruments"]["instruments"]
        self.instrument_to_trade = model_config["model_instruments"]["instrument_to_trade"]
        
        self.kline_to_trade = model_config["model_instruments"]["kline_to_trade"]
        self.volume_to_trade = model_config["model_instruments"]["volume_to_trade"]
        self.tradable_times = model_config["model_instruments"]["tradable_times"]
        self.closing_session_times = model_config["model_instruments"]["closing_session_times"]
        self.futures_trading_window = model_config["model_instruments"]["futures_trading_window"]
        
        
        # UNIVARIATE
        self.signal_function = model_config["signals"]["signal_function"]
        self.z_apply_to = model_config["signals"]["univariate"]["z_apply_to"]
        self.z_window = model_config["signals"]["univariate"]["z_window"]
        self.z_threshold = model_config["signals"]["univariate"]["z_threshold"]
        
        # COMPOSITE
        self.factors = model_config["signals"]["composite"]["factors"]
        self.factors_to_normalise = model_config["signals"]["composite"]["factors_to_normalise"]
        self.factor_loadings = model_config["signals"]["composite"]["factor_loadings"]
        self.scaler_path = model_config["signals"]["composite"]["scaler_path"]
        self.scaler_name =model_config["signals"]["composite"]["scaler_name"]
        self.scaler = pickle_helper.pickle_this(None, pickle_name=self.scaler_name, path=self.scaler_path)
        
        # signal thresholds
        self.normalize_thresholds = model_config["signals"]["thresholds"]["normalize_thresholds"]
        self.L_threshold_lookback = model_config["signals"]["thresholds"]["L_threshold_lookback"]
        self.L_threshold_upper = model_config["signals"]["thresholds"]["L_threshold_upper"]
        self.L_threshold_lower = model_config["signals"]["thresholds"]["L_threshold_lower"]
        self.S_threshold_lookback = model_config["signals"]["thresholds"]["S_threshold_lookback"]
        self.S_threshold_upper = model_config["signals"]["thresholds"]["S_threshold_upper"]
        self.S_threshold_lower = model_config["signals"]["thresholds"]["S_threshold_lower"]
        
        # position sizing 
        self.position_size_ATR_window = model_config["signals"]["position_sizing"]["position_size_ATR_window"]
        self.position_size_norm_window = model_config["signals"]["position_sizing"]["position_size_norm_window"]
        self.position_size_range = model_config["signals"]["position_sizing"]["position_size_range"]
        self.max_holding_period = model_config["signals"]["position_sizing"]["max_holding_period"]
        
        
        
    def calc_signals(self, instruments_dict: dict, verbose=False):
        # importlib.reload(signals)
        # importlib.reload(composite_signals)
        # importlib.reload(pickle_helper)
        
        if verbose: logger.info(f"{'-'*20}\n Calculating Signals \n{'-'*20}\n")

        
        
        
    
            
        # =================================
        # Merge instruments by timeframes
        # =================================     
        sig_dict_prelim = signals.merge_instruments_dict(instruments_dict = instruments_dict,
                                                         spot_instrument = self.spot_instrument,
                                                         futures_instruments = self.futures_instruments,
                                                         futures_trading_times = self.futures_trading_window,
                                                         verbose = False)
        self.sig_dict_prelim = sig_dict_prelim.copy()
        
        
        # =================================
        # z-scores
        # =================================
        sig_dict = signals.populate_z_signals(sig_dict_prelim,
                                              lookback_windows = self.z_window,
                                              sig_threshold = self.z_threshold,
                                              columns = self.z_apply_to,
                                              verbose=False
                                              )
        
        
        # =================================
        # Factor Loadings
        # =================================
        if len(self.factors) > 1:
            logger.info(f"Calculating factor loadings: {self.factors}")
            sig_dict = composite_signals.calc_composite_signal(sig_dict,
                                                               factors_to_compose = self.factors,
                                                               factors_to_normalise = self.factors_to_normalise,
                                                               factor_loadings = self.factor_loadings,
                                                               scaler=self.scaler
                                                               )
        
        self.sig_dict = sig_dict
        
        # =================================
        # Signal Thresholds 
        # =================================
        sig_dict = signal_thresholds.calc_signal_thresholds(sig_dict,
                                                            normalize = self.normalize_thresholds,
                                                            target_signal = self.target_signal,
                                                            L_q_lookback = self.L_threshold_lookback,
                                                            L_q_upper = self.L_threshold_upper,
                                                            L_q_lower = self.L_threshold_lower,
                                                            S_q_lookback = self.S_threshold_lookback,
                                                            S_q_upper = self.S_threshold_upper,
                                                            S_q_lower = self.S_threshold_lower
                                                            )
        
        # =================================
        #  Merge final signals 
        # =================================
        df_sig = composite_signals.merge_signals(sig_dict, smallest_timeframe = self.timeframes[0], target_signal = self.target_signal)
        # self.df_sig = df_sig
        
        
        
        
        # =================================
        #   Position sizing  
        # =================================
        
        # import importlib
        from signals import position_sizing
        # # importlib.reload(position_sizing)
        
        size_timeframe = self.timeframes[0]
        df_sig = position_sizing.calc_position_sizes(df_sig,
                                                     high = f"{size_timeframe}_high",
                                                     low = f"{size_timeframe}_low",
                                                     close = f"{size_timeframe}_close",
                                                     window_ATR = self.position_size_ATR_window,  #14,
                                                     window_norm = self.position_size_norm_window,  #288,
                                                     newRange=self.position_size_range,  #(1,2)
                                                     )
        
        if verbose: logger.info(f"{'-'*20}\n SIGNALS READY \n{'-'*20}\n")
        
        self.df_sig = df_sig            
            
        return df_sig
    
    def rename_signals(self, df_sig):
        # df_trade = df_sig.iloc[:-1,:].copy()
        df_trade = df_sig
        target_signal = self.target_signal
        sig_timeframe = self.smallest_timeframe + "_"
        
        signals = [f"{sig_timeframe}{target_signal}",
                   f"{sig_timeframe}L_uq",
                   f"{sig_timeframe}L_lq", 
                   f"{sig_timeframe}S_uq",
                   f"{sig_timeframe}S_lq"]
        df_trade[["sig", "L_uq", "L_lq", "S_uq", "S_lq"]] = df_trade[signals]
        return df_trade
    
    def insert_tradable_flags(self,df_trade):
        
        # ===========================
        # CHECK TRADABLE TIMES --> to be moved to preprocessing
        # ===========================
        if (len(self.tradable_times) == 0) and (len(self.closing_session_times) == 0):
            df_trade["tradable"] = True
            df_trade["session_closing"] = False
        else:
            assert len(self.tradable_times) == len(self.closing_session_times)
            
            trading_times_index = []
            session_closing_index = []
            for tradable_time, session_closing_time in zip(self.tradable_times, self.closing_session_times):
                trading_times_index += list(df_trade.between_time(tradable_time[0], tradable_time[1]).index)
                session_closing_index += list(df_trade.between_time(session_closing_time[0], session_closing_time[1]).index)
            
            # Add flags to main dataframe
            df_trade["tradable"] = df_trade.index.isin(trading_times_index)
            df_trade["session_closing"] = df_trade.index.isin(session_closing_index)
            
        signals = ["sig", "L_uq", "L_lq", "S_uq", "S_lq"]
        sig_timeframe = self.smallest_timeframe + "_"
        
        last_4_rows = df_trade[[f'{sig_timeframe}open', f'{sig_timeframe}close', f'{sig_timeframe}volume']+signals[0:3]].tail(4)
        return df_trade, last_4_rows
    
    def preprocess(self, instruments_dict: dict, verbose=False, timeit=False):
        
        df_sig = self.calc_signals(instruments_dict, verbose=verbose)
        
        df_trade = self.rename_signals(df_sig)
        
        df_trade,last_4_rows = self.insert_tradable_flags(df_trade)
        return df_trade, last_4_rows