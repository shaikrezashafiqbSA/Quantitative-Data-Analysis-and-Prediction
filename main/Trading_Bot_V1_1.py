import time
import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
# from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler


from data_manager import klines_MR
from models import Mean_Reversion
from order_manager import KGI_order_manager

from signals import signal_snp_momentum
from logger.logger import logger
from backtester import backtest
from telegram_manager.telegram_updater import Telegram_Updater


class Trading_Bot:

    """
    This bot will consume an ml model + market data (transformed into technical indicators) 
    to generate trade signals.

    ml model will be exogenous, whilst market data (transformed into technical indicators)
    
    """
    def __init__(self,
                 model_config: dict,
                 resume = False,
                 ):
        
        # =============================================================================
        # MODEL configurations (to be abstracted into json input)
        # =============================================================================
        self.model_config = model_config
        
        # Paths
        self.output_path = "./telegram/"
        self.output_trades_name = self.model_config["model_name"] + " " + "trades.png"
        self.output_plotly_name = self.model_config["model_name"] + " " + "interactive_charts"
        

        
        # Other instantiations for clarity
        self.df_sig = None 
        self.df_trades = None
        self.df_backtested = None
        self.resume = resume
        
        if self.model_config["run_mode"] == "paper":
            self.TelBot = Telegram_Updater(self.model_config,
                                           load_and_preprocess_klines = self.load_and_preprocess_klines,
                                           resume = self.resume)
        
        
        
    # =============================================================================
    # =============================================================================
    # 1) DATA METHODS
    # =============================================================================
    # =============================================================================
    def load_and_preprocess_klines(self, limit_ccxt=1000, limit_TV=10000):
        """
        Load ALL available klines data from database and make REST api calls to update database
        """
        
        klines_manager_mr = klines_MR.KlinesManagerMR(self.model_config)
        instruments_dict = klines_manager_mr.load_and_update(limit_ccxt=limit_ccxt, limit_TV=limit_TV)
        self.check = instruments_dict
        
        signals_manager_mr = signal_snp_momentum.Signal_Mean_Reversion(self.model_config)
        df_trade,last_4_rows = signals_manager_mr.preprocess(instruments_dict)
        
        # Ensure df_trade is trimmed to current time so that wont look at latest unfinished bar
        t_now = dt.utcnow() - datetime.timedelta(minutes=1)
        df_trade = df_trade[:t_now.strftime("%Y-%m-%d %H:%M:%S")]
        last_4_rows = last_4_rows[:t_now.strftime("%Y-%m-%d %H:%M:%S")].tail(2)
        logger.info(f"\n{'='*20}\nLast 2 rows of df_trade:\n{'='*20}\n{last_4_rows}\n")
        return df_trade
    
    
    def handle_market_data(self, msg):
        pass
    
    def update_klines(self, msg):
        """
        to trim ram data upon receiving new data, and append new data
        """ 
        pass
        
    def load_account(self):
        self.client = KGI_order_manager.KGI_Client(self.model_config)
    
    
    # ==========================================================================================================================================================
    # ==========================================================================================================================================================
    # 2) Start BOT
    # ==========================================================================================================================================================
    # ==========================================================================================================================================================
    def start_bot(self):
        """
        This will be main body of bot where 
        0) checks model_state from transactions_table 
            {"long": {"in_position": True, 
                      "buy": False,
                      "sell": False,
                      "ID": 1
                      }
             "short": {"in_position": False, 
                       "buy": False,
                       "sell": False,
                       "ID": 0
                       }
             }
        1) Load klines data for instruments
        2) read orders from orders table
        3) load positions for subscribed instruments
        4) load strategy state based on loaded positions
        5) **** Check for open positions
        6) 
        """
        # =====================================================================
        # 1) Load model state from client 
        # =====================================================================
        logger.info(f"{'_'*100}\n(0) Running {self.model_config['model_name']} \n{'_'*100}\n\n")
        
        logger.info(f"\n\n{'_'*100}\n(1) Load & Preprocess Klines \n{'_'*100}\n\n")
        if self.model_config["run_mode"] == "paper":
            self.df_trade = self.load_and_preprocess_klines(limit_ccxt=1000, limit_TV=10000)
            self.run_schedulers()
            
        else:
            self.load_account()
            self.df_trade = self.load_and_preprocess_klines(limit_ccxt=1000, limit_TV=10000)
            self.model_state = self.client.load_model_state(model_config = self.model_config)
            
            
            runtime = dt.now().strftime("%Y-%m-%d, %H:%M:%S")
            logger.info(f"\n\n{'_'*100}\nRUNTIME START: {runtime} \n{'_'*100}\nINITIAL MODEL STATE: \n{pd.DataFrame(self.model_state)}\n")
            
            
            # =====================================================================
            # 2) run scheduler based bot
            # =====================================================================
            self.run_schedulers()
        
  
        
    # =============================================================================
    # =============================================================================
    # 1) SCHEDULERS
    # =============================================================================
    # =============================================================================
    def run_schedulers(self):
        # self.scheduler = BackgroundScheduler(timezone="UTC")
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        
        if self.model_config["run_mode"] == "paper":
            # minutes = ','.join([str(int(x)) for x in (np.linspace(0,60,13)) if x<60])
            hours = ','.join([str(int(x)) for x in (np.linspace(0,25,23)) if x<24])
            # TODO: all data should have come in by the 15th second but to check more rigorously
            self.scheduler.add_job(self.TelBot.telegram_update,kwargs={"EOD_report_flag":False}, trigger='cron',second='0',minute="0", hour=hours)
            # self.scheduler.add_job(self.TelBot.telegram_update,kwargs={"EOD_report_flag":True}, trigger='cron',minute='5', hour='10')
        else:
            minutes = ','.join([str(int(x)) for x in np.arange(0,60) if x<60])
            # TODO: all data should have come in by the 15th second but to check more rigorously
            self.scheduler.add_job(self._run_bot_cycle, trigger='cron',minute=minutes,second='0')
            # self.scheduler.add_job(self._run_bot_cycle, trigger='cron', second='0')
            
        self.scheduler.start()
        
        
        
    def _run_bot_cycle(self,):

        # =====================================================================
        # 1) Check model state ie; check for order fills
        # =====================================================================
        runtime_start = dt.now().strftime("%Y-%m-%d, %H:%M:%S")
        t_start = time.time()
        self.model_state = self.client.update_model_state(self.model_state, self.model_config)
        logger.info(f"\n\n{'_'*100}\n(1) BOT CYCLE START: {runtime_start} \n{'_'*100}\n\nINITIAL MODEL STATE: \n{pd.DataFrame(self.model_state)}\n\n")
        
        # =====================================================================
        # 2) Load data
        # =====================================================================
        logger.info(f"\n\n{'_'*100}\n(2) Load & Preprocess Klines \n{'_'*100}\n\n")
        df_trade = self.load_and_preprocess_klines(limit_ccxt=10, limit_TV=10)
        
  
        # =====================================================================
        # Analyse trade signal
        # =====================================================================
        self.model_state = self.client.update_model_state(self.model_state, self.model_config)
        logger.info(f"\n\n{'_'*100}\n(3) Model \n{'_'*100}\n\n")
        
        # Check if any pending orders, if so cancel but only if theres any new signal
        # Check if pending orders have been filled before trading
        model_state = Mean_Reversion.run_mean_reversion_model(df = df_trade,
                                                              model_config = self.model_config,
                                                              model_state = self.model_state, 
                                                              position_sizing_to_trade = False)
        
        
        
        
        
        
        self.model_state = self.analyze_model_state_and_send_orders(model_state, df_trade)
        
        self.df_trade = df_trade
        runtime_end = dt.now().strftime("%Y-%m-%d, %H:%M:%S")
        runtime_s = np.round(time.time()-t_start,3)
        logger.info(f"\n{'_'*100}\n(5) BOT CYCLE END: {runtime_end} ({runtime_s} s)\n{'_'*100}\n\n")

     
    def analyze_model_state_and_send_orders(self, model_state, df_trade):
        if model_state["long"]["signal_event"] or model_state["short"]["signal_event"]:
            logger.info(f"\n\n{'_'*100}\n(4) Model Decision: SIGNAL TRIGGERED \n{'_'*100}\n{pd.DataFrame(model_state)}\n\n")
            model_state = self.client.send_orders(model_state)
        else: 
            logger.info(f"\n\n{'_'*100}\n(4) Model Decision: NONE \n{'_'*100}\n{pd.DataFrame(model_state)}\n\n")
        
        return model_state
            
    def stop(self,):
        self.scheduler.pause()
        # 1) waits for all pending order to be filled then shutdown or cancel immediately and shutdown
        logger.info(f"\n\n{'!'*100}\nSHUTDOWN INITIATED... cancelling all pending orders if have \n{'!'*100}\n")
        
        if self.model_config["run_mode"] == "live":
            logger.info(f"\n{pd.DataFrame(self.model_state)}\n")
            for position in ["long","short"]:
                for side in ["buy","sell"]:
                    self.client.cancel_order_model_state_update(self.model_state, position, side)
        logger.info(f"\n{'!'*100}\n SHUTTING DOWN \n{'!'*100}")
        self.scheduler.shutdown()
        
    def pause(self,):
        self.scheduler.pause()
    
    def resume(self,):
        self.scheduler.resume()
        
        
    def run_backtest(self, publish_plots=False):
        
        df_trade = self.load_and_preprocess_klines()
        df_tested, df_trades, df_metrics = backtest.backtest(df_trade,
                                                             kline_to_trade=self.model_config["model_instruments"]["kline_to_trade"],
                                                             volume_to_trade = self.model_config["model_instruments"]["volume_to_trade"],
                                                             position_sizing_to_trade="size",
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
        
        
        
        
#%%
if __name__ == "__main__":
    
    from config.load_config import load_config_file
    # Load model name from config-dev
    MODEL_NAME = load_config_file('./config/config-dev.json')['model_name']
    # Load model config from given model name in config-dev.json
    model_config = load_config_file(f'./config/{MODEL_NAME}.json')
    
    bot = Trading_Bot(model_config)    
    # df_tested,df_trades,df_metrics = bot.run_backtest(publish_plots=True)
    
    
    bot.start_bot()
    
    #%% signal check 

    #%%

    
    
    #%%
    import ccxt.pro
    import ccxt

    client = ccxt.okx({'verbose':True,
                    "apiKey": '9d3efa18-3752-4151-8821-190235f2ccca',
                    "secret": '5B7387FBF877C91C7D35025255B8E340',
                    "password": '#$yVmSTvdW3L9NWm',
                    'verbose': True,
                    'options': {'defaultType': 'swap',
                                'test':True},
                    # 'headers': {"x-simulated-trading": 1, "OK_ACCESS_KEY":'9d3efa18-3752-4151-8821-190235f2ccca'}
                    })


                        
    

    client.set_sandbox_mode(True)


    #%%
    bal = client.fetchBalance()

    #%%
    markets = client.load_markets()
    

    #%%
    
    # %%
    symbol = 'BTC-USDT'
    type = 'market'  # or 'market'
    side = 'buy'  # or 'sell'
    amount = 0.0001
    price = 30000  # or None

    order = client.create_order(symbol, type, side, amount, price)
    print(order)

    # %%
    # import requests
    # import json
    # import time
    # import hashlib
    # import base64

    # api_key = '9d3efa18-3752-4151-8821-190235f2ccca'
    # secret_key = '5B7387FBF877C91C7D35025255B8E340'
    # passphrase = '#$yVmSTvdW3L9NWm'

    # timestamp = str(time.time())
    # method = 'POST'
    # request_path = '/api/v5/trade/order'

    # body = {
    #     "instId": "BTC-USDT",
    #     "tdMode": "cash",
    #     "side": "buy",
    #     "ordType": "market",
    #     "sz": "0.0001"
    # }

    # body_json = json.dumps(body)

    # message = timestamp + method + request_path + body_json
    # message = message.encode('utf-8')
    # # secret_key = base64.b64decode(secret_key)
    # signature = hashlib.sha256(message + secret_key).hexdigest()

    # url = 'https://okx.com' + request_path

    # headers = {
    #     'Content-Type': 'application/json',
    #     'OK-ACCESS-KEY': api_key,
    #     'OK-ACCESS-SIGN': signature,
    #     'OK-ACCESS-TIMESTAMP': timestamp,
    #     'OK-ACCESS-PASSPHRASE': passphrase,
    # }

    # response = requests.post(url, headers=headers, data=body_json)

    # print(response.json())


    # %% Checking account balanc
    api_key = '9d3efa18-3752-4151-8821-190235f2ccca'
    secret_key = '5B7387FBF877C91C7D35025255B8E340'
    passphrase = '#$yVmSTvdW3L9NWm'

    import okx.Account as Account

    flag = "1"
    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    result = accountAPI.get_account_balance()
    print(result)
    # %% Checking account config
    result = accountAPI.get_account_config()
    if result["code"] == "0":
        acctlv = result["data"][0]["acctLv"]
        if acctlv == "1":
            print("Simple Mode")
        elif acctlv == "2":
            print("Single currency margin mode")
        elif acctlv == "3":
            print("Multi-currency margin mode")
        elif acctlv == "4":
            print("Portfolio margin mode")

    #%% Get available trading pairs
    import okx.PublicData as PublicData

    flag = "1"  # live trading: 0, demo trading: 1

    PublicDataAPI = PublicData.PublicAPI(flag=flag)

    result = PublicDataAPI.get_instruments(
    instType="SWAP"
    )
    res= pd.DataFrame(result['data'])


    # %% Placing orders
    import okx.Trade as Trade
    tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
    instId = "BTC-USDT-SWAP"
    result = tradeAPI.place_order(
        instId=instId,
        tdMode="cross",
        side = "sell",
        ordType = "limit",
        px = "30058.5",
        sz="1"
    )

    if result["code"] == "0":
        print(f"Successful order request: {result['data'][0]['ordId']}")
    else:
        print(f"Unsuccessful order request: {result}")

    # %% Placing orders MARKET
    import okx.Trade as Trade
    tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
    instId = "BTC-USDT-SWAP"
    result = tradeAPI.place_order(
        instId=instId,
        tdMode="cross",
        side = "buy",
        ordType = "market",
        sz="10"
    )

    if result["code"] == "0":
        print(f"Successful order request: {result['data'][0]['ordId']}")
    else:
        print(f"Unsuccessful order request: {result}")

# %% check order status

result = tradeAPI.get_order(instId=instId, ordId=result['data'][0]['ordId'])
print(result)


# %% cancel order

result = tradeAPI.cancel_order(instId=instId, ordId = result['data'][0]['ordId'])
print(result)


# %% get all open orders
all_open_orders_list = tradeAPI.get_order_list()



# %%
