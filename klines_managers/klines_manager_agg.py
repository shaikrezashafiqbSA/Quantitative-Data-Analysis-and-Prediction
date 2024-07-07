
import pandas as pd

from klines_managers import klines_ccxt, klines_polygon
from klines_managers.klines_barchart import load_ohlcv as load_ohlcv_barchart
from klines_managers.klines_KGI import load_ohlcv as load_ohlcv_KGI
from klines_managers.klines_meta import load_ohlcvs as load_ohlcvs_metatrader
from utils.pickle_helper import pickle_this
class KlinesManagerAgg():
    def __init__(self,
                 instruments = ["polygon__C:USDSGD", "polygon__C:USDJPY", "polygon_C:USDGBP", "CCXT_binance__BTCUSD", "KGI_USDSGD"],
                 timeframes = ["1m"],
                 since = "2020-01-01 00:00:00",
                 limits = {"ccxt":1000,"TV":10000,"polygon":5000},
                 update = True):
        self.instruments = instruments
        self.timeframes = timeframes
        self.since = since
        self.limits = limits
        self.update = update
        
    
    def load_ohlcvs(self, update = False):
        #TODO: clean up this function
        # Search through instruments for ccxt vs TV source of data
        instruments_KGI_list = []
        instruments_TV_list = []
        instruments_ccxt_list = []     
        instruments_polygon_list = []   
        instruments_metatrader_list = []
        instruments_barchart_list = []
        for instrument in self.instruments:
            print(instrument)
            exchange,symbol = instrument.split("__")
            if 'ccxt' in exchange:
                instruments_ccxt_list.append(instrument)
            elif "polygon" in exchange:
                instruments_polygon_list.append(symbol)
            elif "metatrader" in exchange:
                instruments_metatrader_list.append(symbol)
            elif "KGI" in exchange:
                instruments_KGI_list.append(symbol)
            elif "barchart" in exchange:
                instruments_barchart_list.append(symbol)
            else:
                instruments_TV_list.append(symbol)
  
        instruments_dict_ccxt = {}
        if len(instruments_ccxt_list) > 0:
            print(f"CCXT LOADING: {instruments_ccxt_list}")
            kline_manager_ccxt = klines_ccxt.KlinesManagerCCXT()
            instruments_dict_ccxt = kline_manager_ccxt.load_ohlcvs(instruments = instruments_ccxt_list,
                                                                   timeframes = self.timeframes,
                                                                   since = self.since,
                                                                   limit = self.limits["ccxt"],
                                                                   update=update)
        
        instruments_dict_polygon = {}
        if len(instruments_polygon_list) > 0:
            print(f"POLYGON LOADING: {instruments_polygon_list}")
            kline_manager_polygon = klines_polygon.KlinesManagerPolygon()
            instruments_dict_polygon = kline_manager_polygon.load_ohlcvs(instruments = instruments_polygon_list,
                                                                    timeframes = self.timeframes,
                                                                    since = self.since,
                                                                    limit = self.limits["polygon"],
                                                                    update=update)
        instruments_dict_metatrader = {}
        if len(instruments_metatrader_list) > 0:
            print(f"METATRADER LOADING: {instruments_metatrader_list}")
            # DATA AVAILABILITY: 2015-01-02 ---> 2023-10-31
            instruments_dict_metatrader = load_ohlcvs_metatrader(instruments = instruments_metatrader_list,
                                                                timeframes = self.timeframes,
                                                                since = "2020-01-01 00:00:00",
                                                                limit = 1000,
                                                                update=False, 
                                                                path = "./database/klines/metatrader/")
            # load_ohlcv_metatrader(update=True, path = "./database/klines/metatrader/USDSGD_1m")
            # df = pickle_this(data=None, pickle_name="USDSGD_1m", path="./database/klines/metatrader/")
            # temp = {}
            # temp["1m"] = df[self.since:]
            # instruments_dict_metatrader[instruments_metatrader_list[0]] = temp

        instruments_dict_KGI = {}
        if len(instruments_KGI_list) > 0:
            print(f"KGI LOADING: {instruments_KGI_list}")
            df_KGI = load_ohlcv_KGI(table_name = 'fx_aggregate_1m', 
                                    instrument= "USD/SGD",
                                    #since="2023-11-17",
                                    # to="2023-11-18"
                                    )
            

            # df_metatrader_barchart = pickle_this(pickle_name = "USDSGD_1m",path= "./database/klines/")
            # df_metatrader_barchart_KGI = pd.concat([df_metatrader_barchart, df_KGI])
            # df_metatrader_barchart_KGI.sort_index(inplace=True)

            temp = {}
            temp["1m"] = df_KGI
            instruments_dict_KGI[instruments_KGI_list[0]] = temp
            # print(instruments_dict_KGI)

        instruments_dict_barchart = {}
        if len(instruments_barchart_list) > 0:
            print(f"BARCHART LOADING: {instruments_barchart_list}")
            df_barchart = load_ohlcv_barchart(update = False)
            temp = {}
            temp["1m"] = df_barchart
            instruments_dict_barchart[instruments_barchart_list[0]] = temp
            # print(instruments_dict_barchart)

        instruments_dict_TV = {}
        # if len(instruments_TV_list) > 0:
        #     print(f"TV LOADING: {instruments_TV_list}")
        #     klines_manager_TV = klines_TV.KlinesManagerTV()
        #     instruments_dict_TV = klines_manager_TV.load_ohlcvs(instruments = instruments_TV_list,
        #                                                         timeframes = self.timeframes,
        #                                                         limit = self.limits["TV"],
        #                                                         update=self.update)
        instruments_dict = instruments_dict_TV | instruments_dict_ccxt | instruments_dict_polygon | instruments_dict_metatrader | instruments_dict_KGI | instruments_dict_barchart
        
        return instruments_dict
            
        
