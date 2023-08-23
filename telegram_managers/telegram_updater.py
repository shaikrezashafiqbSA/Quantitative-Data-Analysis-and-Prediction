import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

from data_manager import db_manager 
from backtester import backtest
from performance import plotly_studies

import telegram
import dataframe_image as dfi 

class Telegram_Updater:
    def __init__(self,
                 model_config:dict,
                 load_and_preprocess_klines,resume=False
                 ):
        self.resume = resume
        self.model_config = model_config
        self.model_name = model_config["model_name"]
        self.df_tested = pd.DataFrame()
        self.df_trades = pd.DataFrame()
        self.df_metrics = pd.DataFrame()
        self.trades_done_today = 0
        self.load_and_preprocess_klines = load_and_preprocess_klines
        
        
        self.output_path = self.model_config["telegram_settings"]["output_path"]
        self.output_trades_name = self.model_config["model_name"] + "_" + "trades.png"
        self.output_plotly_name = self.model_config["model_name"] + "_" + "interactive_charts"
        
    def load_transactions_table():
        transactions_df = db_manager.read_table(table_name="transactions")
        transactions_df = transactions_df[transactions_df["model_name"]==model_config["model_name"]]
        
        
    def run_backtest(self, df_trade, publish_plots=False):
        df_tested, df_trades, df_metrics = backtest.backtest(df_trade,
                                                             kline_to_trade=self.model_config["model_instruments"]["kline_to_trade"],
                                                             volume_to_trade = self.model_config["model_instruments"]["volume_to_trade"],
                                                             position_sizing_to_trade=None,
                                                             tradable_times = self.model_config["model_instruments"]["tradable_times"],
                                                             closing_session_times = self.model_config["model_instruments"]["closing_session_times"],
                                                             signal_function = self.model_config["signals"]["signal_function"],
                                                             max_holding_period = self.model_config["signals"]["position_sizing"]["max_holding_period"],
                                                             fee = self.model_config["equity_settings"]["fee"],
                                                             slippage = self.model_config["equity_settings"]["slippage"],
                                                             long_equity = self.model_config["equity_settings"]["long_allocation"],
                                                             short_equity = self.model_config["equity_settings"]["short_allocation"],
                                                             long_notional = self.model_config["equity_settings"]["long_trade_notional"],
                                                             short_notional = self.model_config["equity_settings"]["short_trade_notional"],
                                                             figsize=(15,10),  # width, height
                                                             show_B=False,
                                                             title= self.model_config["model_name"],
                                                             plots=publish_plots,
                                                             diagnostics_verbose=False)
        

        
        return df_tested,df_trades,df_metrics
    
    
    def generate_plots(self, df_tested, window=["2022-01-01","2023-12-31"], filename_prefix="daily"):
        # importlib.reload(plotly_studies)
        
        
        timeframe = self.model_config["model_instruments"]["timeframe_to_trade"] # this could be an issue if timeframes are not in ascending order
        factor_instruments = self.model_config["model_instruments"]["factor_instruments"]
        futures_instruments = self.model_config["model_instruments"]["futures_instruments"]
        spot_instrument = self.model_config["model_instruments"]["spot_instrument"]
        output_path = self.model_config
        # Check if need to plot futures or factor 
        if len(factor_instruments) == 0 and len(futures_instruments) > 0:
            klines_plots_labels = [f"{spot_instrument} and {futures_instruments[0]}"]
            row_heights = [1,3,1,1]
            subplot_titles = ["Long/short/total cumulative PNL $"] + klines_plots_labels + ["signal", 'size']#"ES1!"]
            
        elif len(factor_instruments) == 0 or len(futures_instruments) == 0:

            
            factor_instrument_name = factor_instruments[0].split("__")[-1]
            ohlcf = {"instrument":factor_instruments[0],
                    "open":f"{timeframe}_open_{factor_instrument_name}",
                    "high":f"{timeframe}_high_{factor_instrument_name}",
                    "low":f"{timeframe}_low_{factor_instrument_name}",
                    "close":f"{timeframe}_close_{factor_instrument_name}",
                    "up_color":'rgb(14,203,129)',
                    "down_color":'rgb(233,67,89)',
                    "opacity":1}
            
            ohlc = {"instrument":spot_instrument,
                    "open":f"{timeframe}_open",
                    "high":f"{timeframe}_high",
                    "low":f"{timeframe}_low",
                    "close":f"{timeframe}_close",
                    "up_color":'rgb(14,203,129)',
                    "down_color":'rgb(233,67,89)',
                    "opacity":1}
            
            
            cols_to_plot = [[ohlcf],
                            [ohlc]+['L_entry_price','L_exit_price','S_entry_price','S_exit_price'],
                            ["cum_A_pnl", "cum_S_pnl","cum_L_pnl"],
                            ['sig__scatter'],
                            ]
            
            klines_plots_labels = [f"{spot_instrument.split('__')[-1]}"]
            row_heights = [2,3,1,1]
            subplot_titles = ["SPX"] + klines_plots_labels + ["PNL", 'signal']#"ES1!"]
            
            
        else:
            klines_plots_labels = factor_instruments + [f"{spot_instrument} and {futures_instruments}"]
            row_heights = [1,1,3,1,1]
        
            ohlc = {"instrument":spot_instrument,
                    "open":f"{timeframe}_open",
                    "high":f"{timeframe}_high",
                    "low":f"{timeframe}_low",
                    "close":f"{timeframe}_close",
                    "up_color":'rgb(14,203,129)',
                    "down_color":'rgb(233,67,89)',
                    "opacity":0.25}
            
            symbol = futures_instruments[0].split("__")[1]
            # TODO: to generalise for more than 1 futures instruments
            ohlc_fut = {"instrument": futures_instruments[0],
                        "open":f"{timeframe}_open_{symbol}",
                        "high":f"{timeframe}_high_{symbol}",
                        "low":f"{timeframe}_low_{symbol}",
                        "close":f"{timeframe}_close_{symbol}",
                        "up_color":'rgb(14, 227, 213)',
                        "down_color":'rgb(201, 105, 20)',
                        "opacity":1}
            
            symbol = factor_instruments[0]#.split("__")[1]
            # TODO: to generalise for more than 1 factor instruments
            ohlc_factor = {"instrument": factor_instruments[0],
                            "open":f"{timeframe}_open_{symbol}",
                            "high":f"{timeframe}_high_{symbol}",
                            "low":f"{timeframe}_low_{symbol}",
                            "close":f"{timeframe}_close_{symbol}",
                            "up_color":'rgb(14,203,129)',
                            "down_color":'rgb(233,67,89)',
                            "opacity":1}
            
            
            cols_to_plot = [["cum_A_pnl", "cum_S_pnl","cum_L_pnl"],
                            [ohlc_factor],
                            [ohlc, ohlc_fut]+['L_entry_price','L_exit_price','S_entry_price','S_exit_price'],
                            ['sig__scatter',"L_lq","L_uq","S_lq","S_uq"],
                            'size',
                            ]
            
            
            
            
            
            subplot_titles = ["Long/short/total cumulative PNL $"] + klines_plots_labels + ["signal", 'size']#"ES1!"]
        
        ps = plotly_studies.build(cols_to_plot = cols_to_plot,
                                  row_heights = row_heights,
                                  height = 1000,
                                  width = 1800,
                                  resampler = False,
                                  publish = True,
                                  output_path = self.output_path,
                                  output_name = filename_prefix + " " + self.output_plotly_name,  # "TFEX_USD interactive_charts"
                                  subplot_titles = subplot_titles)
        ps.plot(df_tested[window[0]:window[1]])
        
    async def telegram_update(self,
                        EOD_report_flag = False,
                        send_to_telegram = True,
                        check_df = True,
                        to_update_flag = False,
                        plots=False): # "EOB"):
        datetimenow = dt.now()
        today = str(datetimenow.date())
        past_24h = str((datetimenow - timedelta(days=1)).date())
        datetimenow_str = datetimenow.strftime("%Y-%m-%d %H:%M:%S")
        datetime_last_update = datetimenow# - timedelta(minutes=10)
        datetime_last_update= datetime_last_update.strftime("%Y-%m-%d %H:%M:%S")
        
        t1 = time.time()
        
        import warnings
        t00 = time.time()
        print(f"\n{'='*40}\n{datetimenow_str} --> START \nlast update: {datetime_last_update}\n{'='*40}\n")
        
        # =============================================================================
        # INITIALISE TELEGRAM CHATBOT
        # =============================================================================
        if send_to_telegram:
            telegram_auth_token = self.model_config["telegram_settings"]["telegram_auth_token"]
            chat_id = self.model_config["telegram_settings"]["chat_id"]
            bot = telegram.Bot(telegram_auth_token)
            
            # msg = f"Signal job starting..."
            # bot.send_message(text=msg, chat_id=chat_id)
            
            
        # =============================================================================
        # Run real-time backtest for trades done
        # =============================================================================
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_trade = self.load_and_preprocess_klines(limit_ccxt=10, limit_TV=10)
            df_tested,df_trades,df_metrics = self.run_backtest(df_trade,publish_plots = False)
            
            self.df_tested = df_tested
            self.df_trades = df_trades
            self.df_metrics = df_metrics
        
        
        print(f"\n\n{'==='*10}\nTELEGRAM UPDATES\n{'==='*10}")
        # =============================================================================
        # CHECK FOR NEW TRADES
        # =============================================================================
        if len(df_trades[today:today])>self.trades_done_today:
            if self.resume:
                to_update_flag = False
                self.resume = False
            else:
                to_update_flag = True
            self.trades_done_today = len(df_trades[today:today])
    
            
        dfi.export(df_trades[today:today], f"{self.output_path}{self.output_trades_name}",table_conversion='matplotlib')
        
    
    
    
        # =============================================================================
        # POST TO TELEGRAM EOB
        # =============================================================================
        if not EOD_report_flag:
            if to_update_flag:
                if send_to_telegram:
                    order_details = self.get_order_details(df_trades)
                    caption = f"{self.model_name} ALERT:\n Signal triggered at {df_trades[today:today].index[-1]} bar\n{order_details}"
                    await bot.send_photo(chat_id, photo=open(f'{self.output_path}{self.output_trades_name}', 'rb'),caption=caption)
                    
                    # self.generate_plots(df_tested, window = [today,today], filename_prefix="Daily")
                    # file = open(f"{self.output_path}Daily {self.output_plotly_name}.html",'rb')
                    # await bot.send_document(chat_id, file)
            else: # if not update flag just update plot for local inspection
                self.generate_plots(df_tested, window = [past_24h,today], filename_prefix="Daily")
                
                
        # =============================================================================
        #  POST TO TELEGRAM EOD
        # =============================================================================
        elif EOD_report_flag:
            self.generate_plots(df_tested, window = [today,today], filename_prefix="Daily")
            if send_to_telegram:
                if len(df_trades[today:today])==0:
                    msg = f"END OF SESSION - No trades done today"
                    await bot.send_message(text=msg, chat_id=chat_id)
                else:
                    # bot.send_photo(chat_id, photo=open(f'{self.output_path}{self.output_trades_name}', 'rb'),caption=f"{today} - futures signal V2 tradelogs")
                    await bot.send_photo(chat_id, photo=open(f'{self.output_path}{self.output_trades_name}', 'rb'),caption=f"{self.model_name} {today} EOD report")
    
    
                file = open(f"{self.output_path}Daily {self.output_plotly_name}.html",'rb')
                await bot.send_document(chat_id, file)
        else:
            self.generate_plots(df_tested, window = [past_24h,today], filename_prefix="Daily")
                
                
                
                
        t2 = np.round(time.time()-t00,2)
        print(f"Signal job completed in {t2}s ({t1}s)")
        datetime_end = dt.now()
        datetimen_end_str = datetime_end.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*40}\n{datetimen_end_str} --> END\n{'='*40}\n")
        if check_df:     
            return df_tested, df_trades, df_metrics
    

    def get_order_details(self, df_trades):
        #  ascertain if long or short
        last_trade = df_trades.iloc[-1,:]
        order_details = []
        for p in ["S","L"]:
            if p == "S":
                position = "short"
            else:
                position = "long"
            if not np.isnan(last_trade[f"{p}_id"]):
                if not np.isnan(last_trade[f"{p}_entry_price"]):
                    side = "entry"
                    price = last_trade[f"{p}_entry_price"]
                    qty = last_trade[f"{p}_cost"]
                elif not np.isnan(last_trade[f"{p}_exit_price"]):
                    side = "exit"
                    price = last_trade[f"{p}_exit_price"]
                    qty = last_trade[f"{p}_cost"]
                order_details.append(f"{position} {side}\nprice: {price}\nqty: {qty}")
        order_details = f"\n\n".join(order_details)
        print(order_details)
        return order_details
#%%
if __name__ == "__main__":
    from data_manager import db_manager 
    
    from config.load_config import load_config_file
    # Load model name from config-dev
    MODEL_NAME = load_config_file('./config/config-dev.json')['model_name']
    # Load model config from given model name in config-dev.json
    model_config = load_config_file(f'./config/{MODEL_NAME}.json')
    
    
    #%%

    #%%

    

