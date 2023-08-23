import numpy as np
import pandas as pd
import pandas_ta
import pandas_ta as ta
import warnings

from signals.indicators import calc_tides,calc_slopes, calc_mfi, calc_emas, calc_rsis, calc_z_sig, calc_kdr

class indicators_manager:
    """
    This class is responsible for calculating technical indicators (using pandas_ta)
    This class also inherits kline/indicators processing from strategy class object, but why? 
    
    
    """
    def __init__(self,
                 indicators: dict = {"hma":{'length':[20,40,80]}}, # "col_names" : ("ADX","DMP","DMN") for multiple outputs
                 postprocess_klines = None,
                 preprocess_klines = None
                 ):
        self.indicators = indicators
        self.indicators_list = pd.DataFrame().ta.indicators(as_list=True)
        if postprocess_klines is not None:
            self.postprocess_klines = postprocess_klines
        if preprocess_klines is not None:
            self.preprocess_klines = preprocess_klines
        # Build list according to what ta.Strategy accepts
        indicators_params = []
        for indicator,params in indicators.items():
            if indicator in ["tide","tide_fast","tide_slow","slopes", "mfi", "ema", "rsi", "z_sig", "kdr"]:
                continue
            kind = indicator
            if len(params) == 1:
                for param_name,param_value_list in params.items():
                    for param_value in param_value_list:
                        indicators_params.append({"kind":kind, 
                                                  param_name:param_value}
                                                 )
            else: 
                try:
                    df = pd.DataFrame(params)
                except:
                    max_param_len = max([len(param) for param in params.values() ])
                    for param_name,param in params.items():
                        if len(param) < max_param_len:
                            params[param_name] = param*max_param_len
                    df = pd.DataFrame(params)
                df["kind"]=kind
                for idx in range(len(df)):
                    ind_dict_i ={}
                    for indicator_label in df.columns:
                        ind_dict_i[indicator_label] = df[indicator_label].iloc[idx]
                    indicators_params.append(ind_dict_i)
       
        self.indicators_factory = ta.Strategy(name="trading_bot",ta=indicators_params)
        self.base_timeframe = None
        # self.base_timeframe_preproces = None
        
    def freq_map(self,freq="4h",preprocess=False):
        #  returns int multiplier for higher timeframes
        # eg: if base timeframe is 1h, then if freq given is 4h, return 4
        freq_number = int(freq[:-1])
        timeframe = freq[-1]
        
        # assert timeframe == "h"
        # if preprocess:
        return int(freq_number/int(self.base_timeframe[:-1]))
        # else:
        #     return int(freq_number/int(self.base_timeframe_preproces[:-1]))
        
        
        
        
    def _calc_indicators(self,klines: pd.DataFrame,freq="1h",workers=0):
        # if called first then take that freq as base timeframe
        if self.base_timeframe is None:
            self.base_timeframe = freq
        if "kdr" in self.indicators.keys():
            klines =  calc_kdr(klines)
            
            
            
        if "z_sig" in self.indicators.keys():
            lookback_bars = self.indicators["z_sig"]["lookback_bars"]
            price = self.indicators["z_sig"]["price"]
            threshold = self.indicators["z_sig"]["threshold"]
            
            klines =  calc_z_sig(klines, lookback_bars=lookback_bars, cols=price, threshold=threshold)
        
        
        if "tide" in self.indicators.keys():
            sensitivity = self.indicators["tide"]["sensitivity"]
            thresholds = self.indicators["tide"]["thresholds"]
            windows = self.indicators["tide"]["window"]
            # print(f"\n\n sensitivity: {sensitivity}, type: {type(sensitivity)}")
            if len(sensitivity) >1 or type(sensitivity) is not int:
                # warnings.warn(f"More than 1 sensitivity parameter not supported yet \n--> selecting 1st of {sensitivity}")
                sensitivity = int(sensitivity[0])
            if len(thresholds) >1 or type(thresholds) is not int:
                # warnings.warn(f"More than 1 threshold parameter not supported yet \n--> selecting 1st of {thresholds}")     
                thresholds = int(thresholds[0])
            if type(windows) is not np.ndarray:
                windows = np.array(windows)*self.freq_map(freq)
                # print(f"{windows}, type: {type(windows[0])}")
            price = self.indicators["tide"]["price"]  

            klines = calc_tides(klines,sensitivity=sensitivity, thresholds=thresholds, windows=windows, price=price, suffix="")


        if "tide_slow" in self.indicators.keys():
            sensitivity = self.indicators["tide_slow"]["sensitivity"]
            thresholds = self.indicators["tide_slow"]["thresholds"]
            windows = self.indicators["tide_slow"]["window"]
            # print(f"\n\n sensitivity: {sensitivity}, type: {type(sensitivity)}")
            if len(sensitivity) >1 or type(sensitivity) is not int:
                # warnings.warn(f"More than 1 sensitivity parameter not supported yet \n--> selecting 1st of {sensitivity}")
                sensitivity = int(sensitivity[0])
            if len(thresholds) >1 or type(thresholds) is not int:
                # warnings.warn(f"More than 1 threshold parameter not supported yet \n--> selecting 1st of {thresholds}")     
                thresholds = int(thresholds[0])
            if type(windows) is not np.ndarray:
                windows = np.array(windows)*self.freq_map(freq)
                # print(f"{windows}, type: {type(windows[0])}")
            price = self.indicators["tide_slow"]["price"]  

            klines = calc_tides(klines,sensitivity=sensitivity, thresholds=thresholds, windows=windows, col=price, suffix="slow")
            
            
        if "tide_fast" in self.indicators.keys():
            sensitivity = self.indicators["tide_fast"]["sensitivity"]
            thresholds = self.indicators["tide_fast"]["thresholds"]
            windows = self.indicators["tide_fast"]["window"]
            # print(f"\n\n sensitivity: {sensitivity}, type: {type(sensitivity)}")
            if len(sensitivity) >1 or type(sensitivity) is not int:
                # warnings.warn(f"More than 1 sensitivity parameter not supported yet \n--> selecting 1st of {sensitivity}")
                sensitivity = int(sensitivity[0])
            if len(thresholds) >1 or type(thresholds) is not int:
                # warnings.warn(f"More than 1 threshold parameter not supported yet \n--> selecting 1st of {thresholds}")     
                thresholds = int(thresholds[0])
            if type(windows) is not np.ndarray:
                windows = np.array(windows)*self.freq_map(freq)
                # print(f"{windows}, type: {type(windows[0])}")
            price = self.indicators["tide_fast"]["price"]  

            klines = calc_tides(klines,sensitivity=sensitivity, thresholds=thresholds, windows=windows,price=price, suffix="")
            
        if "slopes" in self.indicators.keys():
            slope_lengths = self.indicators["slopes"]["slope_lengths"]
            scaling_factor = self.indicators["slopes"]["scaling_factor"]
            lookback = self.indicators["slopes"]["lookback"]
            upper_quantile = self.indicators["slopes"]["upper_quantile"]
            logRet_norm_window = self.indicators["slopes"]["logRet_norm_window"]
            
            for sf in scaling_factor:
                for lb in lookback:
                    for uq in upper_quantile:
                        for lw in logRet_norm_window:
                            suffix = f"{lb}_{lw}"
                            # ensure all int or float
                            klines = calc_slopes(klines, 
                                                 slope_lengths=slope_lengths,
                                                 scaling_factor=sf, 
                                                 lookback=lb, 
                                                 upper_quantile=uq,
                                                 logRet_norm_window=lw, 
                                                 suffix=suffix)
                            
        if "mfi" in self.indicators.keys(): #and not self.indicators["mfi"]["preprocess"]:
            lengths = self.indicators["mfi"]["length"]
            for length in lengths:
               window = length * self.freq_map(freq)
               klines = calc_mfi(klines, window,label=length)
               
        if "rsi" in self.indicators.keys():
            lengths = self.indicators["rsi"]["length"]
            prices = self.indicators["rsi"]["price"]
            for length in lengths:
                for price in prices:
                   window = length * self.freq_map(freq)
                   klines = calc_rsis(klines, price, window,label=f"{length}_{price[0]}")
               
        if "ema" in self.indicators.keys():
            lengths = self.indicators["ema"]["length"]
            for length in lengths:
               window = length * self.freq_map(freq)
               for price in self.indicators["ema"]["price"]:
                   klines = calc_emas(klines,price, window,label=length)
                   
        if "psar" in self.indicators.keys():
            psar = klines.ta.psar()
            l=psar.filter(regex="PSARl").columns[0]
            s=psar.filter(regex="PSARs").columns[0]
            psar=psar[[l,s]]
            psar = psar.fillna(0)
            klines["PSAR"]=psar.sum(axis=1)
        
        klines.ta.cores = workers
        klines.ta.strategy(self.indicators_factory,verbose=False) 
        
        return klines
        
    def _preprocess_klines(self,klines: pd.DataFrame, freq="1h"):
        # if self.base_timeframe_preproces is None:
            # self.base_timeframe_preproces = freq
        # """
        # This allows features to be processed further (generate MFI then run tide on MFIs)
        # """
        # if "mfi" in self.indicators.keys() and self.indicators["mfi"]["preprocess"]:
        #     lengths = self.indicators["mfi"]["length"]
        #     ohlc = self.indicators["mfi"]["ohlc"]
        #     for length in lengths:
        #        window = length * self.freq_map(freq)
        #        klines = calc_mfi(klines, window,label=length,ohlc=ohlc)
        
        return klines
   
    
    def _postprocess_klines(self,klines: pd.DataFrame, freq="1h"):
        return klines
    # def onPayload(self,payload):
    #     # Calculate TA stuff here
    #     if payload["type"] == "klines":
    #         return self.klinesHandler(payload)

        
        
        
        
#%%
    # def calc_indicator(klines,
    #                    indicator="LINEARREG_SLOPE",
    #                    params={'timeperiod':[20,40,80]}
    #                    ,
    #                    ):
    #     # indicator_factory = vbt.pandas_ta(indicator) # pandas_ta too slow
    #     # catalogue_talib = talib.get_functions()
    #     # try:
    #     #     indicator_factory = vbt.talib(indicator)
    #     # except Exception as e:
    #     #     # print(f"{e} - trying pandas_ta")
    #     indicator_factory = vbt.talib(indicator)
            
    #      #Else if not in pandas or talib then use jeremy's nb indicators
    #     # if custom:
    #     #     indicator_factory = IndicatorFactory(
    #     #                                             class_name='HMA',
    #     #                                             module_name=__name__,
    #     #                                             short_name='hma',
    #     #                                             input_names=['close'],
    #     #                                             param_names=['winsize'],
    #     #                                             output_names=['hma']
    #     #                                         ).from_apply_func(
    #     #                                             hma_nb,
    #     #                                             kwargs_to_args=None,
    #     #                                             ewm=False,
    #     #                                             adjust=False
    #     #                                         )
            
            
    #     output_names = indicator_factory.output_names
    #     input_names = indicator_factory.input_names
    #     param_names = indicator_factory.param_names
        
    #     input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
    #     for input_name in input_names:
    #         input_arguments[input_name]=klines[input_name]
    #     for param_name in param_names:
    #         try:
    #             input_arguments[param_name]=params[param_name]   
    #         except:
    #             # print(f"{indicator} parameter: {param_name} not provided, using default values")
    #             pass
    #     if len(output_names) == 1:
    #         df = getattr(indicator_factory.run(**input_arguments),output_names[0]) # Gotta fix for multiple outputs?
    #     else: 
    #         dfs=[]
    #         for output_name_idx in range(len(output_names)+1):
    #             dfs.append(getattr(indicator_factory.run(**input_arguments),output_names[0]) )
    #         df = pd.concat(dfs, axis=1)
    
    #     return df
#%%    
# from datetime import datetime
# from config.parser import config_to_dict
# config = config_to_dict("config/config-dev.ini")    
# indicators =config["strategy"]["indicators"]
# test= Indicators_Manager(indicators=indicators)
# klines = klines_LTF.copy()


# t0 = time.time()
# df = test.calc_indicator(klines)
# print(f"{time.time()-t0}")


# test.calc_derived_indicator()

#%%
# self.klines_indicators_dict = self.calc_indicators_for_all_timeframes(klines_dict=self.klines_dict,continuous_resample=continuous_resample,workers=workers)
# self.klines_indicators_dict = self.calc_derived_indicators()
        
#     def calc_derived_indicators(self):
#         klines_indicators_dict = {}
#         for TF,klines_indicators in self.klines_indicators_dict.items():
#             klines_indicators_postprocessed = self._calc_indicators_postprocessed(klines_indicators,TF=TF)
#             klines_indicators_dict[TF] = klines_indicators_postprocessed
        
#         return klines_indicators_dict
# # =============================================================================
# # Calculate and insert indicators into klines dataframes
# # =============================================================================
#     def calc_indicators_for_all_timeframes(self, klines_dict, continuous_resample:bool,workers=0):
#         # klines_dict of pairs with 1m, LTF MTF, HTF
#         # BUILD TECHNICAL INDICATORS (pandas_ta uses multiprocessing across TAs)
#         klines_indicators_dict = {}
#         for TF,klines in klines_dict.items():
#             klines_indicators = klines.copy()
#             klines_indicators.ta.cores = workers
#             klines_indicators.ta.strategy(self.technical_indicators_suite,verbose=False)   
#             # klines_indicators.ta.strategy(technical_indicators_suite) 
            
#             for indicator_params in self.indicators_params:
#                 # To transition to pandas ta suite-like object for multiple TAs? 
#                 indicator = indicator_params["kind"]
#                 if "hma" == indicator:
#                     klines[f"{indicator}"] = hma_nb(klines['close'].values, winsize=indicator_params['length'])
            
#             if TF != "1m":
#                 klines_indicators_dict[TF] = klines_indicators
#             else:
#                 klines_indicators_dict['1m'] = klines_indicators
        
            
    
#         return klines_indicators_dict
    

# # =============================================================================
# # Calculate and insert indicators into klines dataframes
# # =============================================================================
#     def calc_indicators(self, klines, timeframe, workers=0):
#         # klines_dict of pairs with 1m, LTF MTF, HTF
#         # BUILD TECHNICAL INDICATORS (pandas_ta uses multiprocessing across TAs)
#         # klines_indicators_dict = {}
#         # for TF,klines in klines_dict.items():
#         klines_indicators = klines.copy()
#         klines_indicators.ta.cores = workers
#         klines_indicators.ta.strategy(self.technical_indicators_suite,verbose=False)   
#             # klines_indicators.ta.strategy(technical_indicators_suite) 
            
#         for indicator_params in self.indicators_params:
#             # To transition to pandas ta suite-like object for multiple TAs? 
#             indicator = indicator_params["kind"]
#             if "hma" == indicator:
#                 klines[f"{indicator}"] = hma_nb(klines['close'].values, winsize=indicator_params['length'])
        
#         return klines_indicators
#%%

# if __name__ == "__main__":
#     import pandas_ta
#     import numpy as np
#     import pandas as pd
#     import vectorbt as vbt
#     from vectorbt.indicators.factory import IndicatorFactory
#     HMA = IndicatorFactory(
#         class_name='HMA',
#         module_name=__name__,
#         short_name='hma',
#         input_names=['close'],
#         param_names=['winsize'],
#         output_names=['hma']
#     ).from_apply_func(
#         hma_nb,
#         kwargs_to_args=None,
#         ewm=False,
#         adjust=False
#     )
        
        
       
#     klines = klines_1m.copy().head(5000)
    
    
#     timeit slope_talib = vbt.talib("LINEARREG_SLOPE").run(klines_1m["close"],[20]).real
    
#     indicator="EMA"
#     timeperiod=[7,10,14,20,28,40,56,80]
    
    
    
#     # def indicator_generator(klines,indicator,params)
#     generator_ema = vbt.talib("EMA")
#     inputs_required = generator_ema.input_names
#     param_names = generator_ema.param_names
    
#     # dictionary for generator_ema input argument: {"close":klines["close"],"timeperiod":timeperiod}
#     input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
#     for input_required in inputs_required:
#         input_arguments[input_required]=klines[input_required]
#     for param_name in param_names:
#         input_arguments[param_name]=timeperiod
    
    
#     indicators_ema = generator_ema.run(**input_arguments)
#     output_names = generator_ema.output_names
#     get_indicators_ema = getattr(indicators_ema,output_names[0])
    
#     indicators = {"HMA":{'length': [7,10,14,20,28,40,56,80]},
#                   "LINEARREG_SLOPE":{'timeperiod':[7,10,14,20,28,40,56,80]}
#                   }
#     #%%
#     def calc_indicator(klines,
#                        indicator="HMA",
#                        params={'length':[20,40,80]}
#                        ,
#                        ):
#         # indicator_factory = vbt.pandas_ta(indicator) # pandas_ta too slow
#         # catalogue_talib = talib.get_functions()
#         # try:
#         #     indicator_factory = vbt.talib(indicator)
#         # except Exception as e:
#         #     # print(f"{e} - trying pandas_ta")
#         indicator_factory = vbt.pandas_ta(indicator)
            
#          #Else if not in pandas or talib then use jeremy's nb indicators
#         # if custom:
#         #     indicator_factory = IndicatorFactory(
#         #                                             class_name='HMA',
#         #                                             module_name=__name__,
#         #                                             short_name='hma',
#         #                                             input_names=['close'],
#         #                                             param_names=['winsize'],
#         #                                             output_names=['hma']
#         #                                         ).from_apply_func(
#         #                                             hma_nb,
#         #                                             kwargs_to_args=None,
#         #                                             ewm=False,
#         #                                             adjust=False
#         #                                         )
            
            
#         output_names = indicator_factory.output_names
#         input_names = indicator_factory.input_names
#         param_names = indicator_factory.param_names
        
#         input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
#         for input_name in input_names:
#             input_arguments[input_name]=klines[input_name]
#         for param_name in param_names:
#             try:
#                 input_arguments[param_name]=params[param_name]   
#             except:
#                 # print(f"{indicator} parameter: {param_name} not provided, using default values")
#                 pass
#         if len(output_names) == 1:
#             df = getattr(indicator_factory.run(**input_arguments),output_names[0]) # Gotta fix for multiple outputs?
#         else: 
#             dfs=[]
#             for output_name_idx in range(len(output_names)+1):
#                 dfs.append(getattr(indicator_factory.run(**input_arguments),output_names[0]) )
#             df = pd.concat(dfs, axis=1)
    
#         return df
    
#     klines = klines_LTF.copy()
    
#     # timeit calc_indicator(klines)
#     test = calc_indicator(klines)
#     #%%
#     def calc_indicator_pandas_ta(klines):
#         dfs = pd.DataFrame()
#         for param in [20,40,80]:
#             df=klines.ta.ema(param)
#             dfs[param]=df
#         return dfs

#     #%%
    
#     klines = klines_LTF.copy()
#     indicators_params = [{"kind":"hma","length":20}, {"kind":"hma","length":40}, {"kind":"hma","length":80}]
#     import pandas_ta as ta
#     technical_indicators_suite = ta.Strategy(name="ta_suite",ta=indicators_params)
#     klines.ta.cores = 0
#     # timeit klines.ta.strategy(technical_indicators_suite,verbose=False) 

    
    
    