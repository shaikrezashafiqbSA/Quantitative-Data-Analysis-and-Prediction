from data_manager import klines_TV, klines_ccxt, klines_polygon
import time

class KlinesManagerMR:
    def __init__(self, 
                 model_config:dict):
        
        self.update_model_config(model_config)
        self.polygon_api_key = "UsWbvzVunOUY30oSsqzDjuoI1gESkiYU"
    def update_model_config(self, model_config):
        self.instruments = model_config["model_instruments"]["instruments"]
        self.timeframes = model_config["model_instruments"]["timeframes"]
        
    
    def load_and_update(self,rows_limit= 3000, limit_ccxt = 1000, limit_TV = 10000,since="2020-01-01 00:00:00"):
        #TODO: rows_limit to prevent ram from being clogged up with ever expanding dataframes size
        # Search through instruments for ccxt vs TV source of data
        
        instruments_TV_list = []
        instruments_ccxt_list = []     
        instruments_polygon_list = []   
        for instrument in self.instruments:
            exchange,symbol = instrument.split("__")
            if 'ccxt' in exchange:
                instruments_ccxt_list.append(instrument)
            elif "polygon" in exchange:
                instruments_polygon_list.append(instrument)
            else:
                instruments_TV_list.append(instrument)
  
        instruments_dict_ccxt = {}
        if len(instruments_ccxt_list) > 0:
            print(f"CCXT LOADING: {instruments_ccxt_list}")
            kline_manager_ccxt = klines_ccxt.KlinesManagerCCXT()
            instruments_dict_ccxt = kline_manager_ccxt.load_ohlcvs(instruments = instruments_ccxt_list,
                                                                   timeframes = self.timeframes,
                                                                   since = since,
                                                                   limit = limit_ccxt)
        
        instruments_dict_polygon = {}
        if len(instruments_polygon_list) > 0:
            print(f"POLYGON LOADING: {instruments_polygon_list}")
            kline_manager_polygon = klines_polygon.KlinesManagerPolygon(api_key = self.polygon_api_key)
            instruments_dict_polygon = kline_manager_polygon.load_ohlcvs(instruments = instruments_polygon_list,
                                                                   timeframes = self.timeframes,
                                                                   since = since,
                                                                   limit = 5000)
            

        instruments_dict_TV = {}
        if len(instruments_TV_list) > 0:
            print(f"TV LOADING: {instruments_TV_list}")
            klines_manager_TV = klines_TV.KlinesManagerTV()
            instruments_dict_TV = klines_manager_TV.load_ohlcvs(instruments = instruments_TV_list,
                                                                timeframes = self.timeframes,
                                                                limit = limit_TV)
        
        instruments_dict = instruments_dict_TV | instruments_dict_ccxt | instruments_dict_polygon
        
        return instruments_dict
            
        


#%%
if __name__ == "__main__":
    
    instruments_ccxt_list = ["ccxt_kucoin__ETH-USDT", "ccxt_kucoin__BTC-USDT"]
    limit_ccxt = 1000
    
    
    
    kline_manager_ccxt = klines_ccxt.KlinesManagerCCXT()
    instruments_dict_ccxt = kline_manager_ccxt.load_ohlcvs(instruments = instruments_ccxt_list,
                                                           timeframes = ["1h"],
                                                           since = "2019-01-01 00:00:00",
                                                           limit = limit_ccxt)
    
    
    
    #%%
    
    import ccxt
    client = ccxt.kucoin()