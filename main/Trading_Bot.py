from datetime import datetime
import time
import numpy as np
import pandas as pd
import asyncio
import websockets
import json
import ssl
import asyncio
from datetime import datetime, timedelta
import pytz
import traceback


from config.load_config import load_config_file
from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger


from logger.logger import setup_logger
from signal_managers.signal_generator import signal_generator
from backtesters.get_trading_decision import calc_sig
from slack_managers.messenger import send_msg, send_msg_trade_update, send_msg_position_update
from utils.pickle_helper import pickle_this
from utils.get_time import get_dt_times_now

from order_managers.KGI_order_manager import KGI_Client
from order_managers.paper_order_manager import Paper_Order_Manager
from backtesters.get_trading_decision import load_ohlcv
from performance_analytics.plots import get_plotly_univariate as plots

logger = setup_logger(logger_file_name="trading_bot")

class Trading_Bot:
    def __init__(self,
                 model_selection = "USDSGD_2m_MR",
                 config_mode = "config-dev",
                 run_mode = "paper", 
                 save_to_db_flag = "True",
                 reload_model_state = False,
                 graceful_resume = False,
                 bot_timeframe_delay_s = 3,
                 bot_position_update_interval_m = 30,

                 prepare_plots_flag = True,
                 publish_html_plotly_dashboard = True,
                 publish_html_file_path = "./backtests/trading_bot_plotly/",

                 replay = False,
                 replay_window = ["2023-11-25 02:15:00", "2023-11-24 17:59:00"],
                 replay_speed = 10,
                 **kwargs
                 ) -> None:
        

        self.model_selection = model_selection
        self.config_mode = config_mode
        self.save_to_db_flag = save_to_db_flag

        # ============================================================================================================
        # Model state, model config options
        # ============================================================================================================
        self.reload_model_state = reload_model_state
        self.model_config = None
        self.model_state = None
        self.model_states = []


        

        self.model_config = self.load_model_config()
        # ansure self.model_config is not none:
        assert self.model_config is not None, "self.model_config is None ---> PLEASE CHECK CONFIG FILE"

        # set all **kwargs into model_config
        for key, value in kwargs.items():
            self.model_config[key] = value

        if run_mode is not None:
            self.model_config["run_mode"] = run_mode
        logger.info(f"\n{'='*100}\n (-1)(a) model_config loaded: \n{self.model_config}\n{'='*100}\n\n")
        # ============================================================================================================
        # Bot timing options
        # ============================================================================================================
        self.trading_start_time = datetime.strptime(self.model_config["tradable_times"][0][0], '%H:%M').time()
        self.trading_end_time = datetime.strptime(self.model_config["tradable_times"][-1][-1], '%H:%M').time()
        self.graceful_resume = graceful_resume
        self.bot_timeframe_delay_s = bot_timeframe_delay_s
        self.bot_position_update_interval_m = bot_position_update_interval_m

        # ============================================================================================================
        # Plotly publish options
        # ============================================================================================================
        self.prepare_plots_flag =prepare_plots_flag
        self.publish_html_plotly_dashboard = publish_html_plotly_dashboard
        self.html_file_path = publish_html_file_path
        self.html_file_name = f"{self.model_config['instrument_to_trade']}_{self.model_config['model_name']}_{self.model_config['signal_to_trade']}_{self.model_config['timeframe_to_trade']}"
       
        # ============================================================================================================
        # REPLAY options
        # ============================================================================================================
        self.replay = replay
        self.replay_window = replay_window
        self.replay_speed = replay_speed
        # ensure replay_speed can be such that the time it does fastest in 1 cycle is a minimally bot_timeframe_delay_s
        if self.replay_speed:
            if int(self.model_config["timeframe_to_trade"][:-1])*60/self.replay_speed <= self.bot_timeframe_delay_s:
                # set replay speed such that time it takes a cycle is bot_timeframe_delay_s
                self.replay_speed = max(self.bot_timeframe_delay_s, int(self.model_config["timeframe_to_trade"][:-1])*60/self.bot_timeframe_delay_s)
                # logger.info(f"\n\n{'='*10}\n\n bot _init_ - replay_speed: {self.replay_speed}\n\n{'='*10}\n\n")

        # ============================================================================================================
        # Plotly matters
        # ============================================================================================================
        self.row_heights = [3,2,2]
        self.subplot_titles = ["Candles, trades, PNL","signal", "trade logs"]

        ohlc = {"instrument":self.model_config['instrument_to_trade'],
                "open":f"open",
                "high":f"high",
                "low":f"low",
                "close":f"close",
                "up_color":'lightblue',
                "down_color":'rgb(233,67,89)',
                "opacity":1}

        self.cols_to_plot = [#'table__df_summary',
                        ["cum_L_pnl", "cum_S_pnl", ohlc,'L_entry_price','L_exit_price','S_entry_price','S_exit_price'],
                        [f'{self.model_config["signal_to_trade"]}'],
                        'table__df_trades',
                        ]
        
        # ============================================================================================================
        # END ATTRIBUTE ASSIGNMENTS
        # ============================================================================================================

    def load_model_config(self,):
        # ============================================================================================================
        # (-1) LOAD model_config
        # ============================================================================================================
        if self.model_selection is None:
            MODEL_NAME = load_config_file(f'./config/{self.config_mode}.json')['model_name']
        else:
            MODEL_NAME = self.model_selection
        model_config = load_config_file(f'./config/{MODEL_NAME}.json')
        model_config["model_name_to_DB"] = MODEL_NAME

        # Ensure no plots are shown
        model_config.update({'show_plots': False,'show_plots_TP': False,})
        return model_config

    def start_bot(self):
        """
        main function to start the bot
        """
    
        self.order_manager = self.load_order_manager(self.model_config, self.save_to_db_flag)

        self.model_state = self.order_manager.load_model_state()
        # logger.info(f"\n{'='*100}\n (-1)(b) model_state loaded: \n{self.model_state}\n{'='*100}\n\n")
        
        # ============================================================================================================
        # warm up bot
        # ============================================================================================================
        
        if self.replay:
            self.model_config["data_update"] = False
        else:
            self.model_config["data_update"] = True
        
        # wait for "tradable_times" in model_config (model_config["tradable_times"] = [["03:00", "03:15"]]) so wait for 3:00 to start bot
        # so sleep from now till 3:00:
        # no
        # self.trading_start_time = datetime.strptime(self.model_config["tradable_times"][0][0], '%H:%M').time()
        # self.trading_end_time = datetime.strptime(self.model_config["tradable_times"][-1][-1], '%H:%M').time()
        # now sleep till self.trading_start_time ("03:00"):
        now = datetime.now().time()
        if self.trading_start_time.minute != 0:
            minutes_to_next_nth_minute = self.trading_start_time.minute - (now.minute % self.trading_start_time.minute)
        else:
            minutes_to_next_nth_minute = 0
        seconds_to_next_nth_minute = minutes_to_next_nth_minute * 60 - now.second
        sleep_time = seconds_to_next_nth_minute + self.bot_timeframe_delay_s
        if sleep_time < 0:
            sleep_time = 0
        # logger.info(f"Waiting for start of the next {self.trading_start_time}th minute to run the trading bot ... {sleep_time} seconds")
        time.sleep(sleep_time)

        self.warm_up_bot()


        # ============================================================================================================
        # Schedule jobs
        # ============================================================================================================
        # Calculate the number of seconds until the next nth minute
        timeframe = int(self.model_config["timeframe_to_trade"][:-1])
        now = datetime.now()
        minutes_to_next_nth_minute = timeframe - (now.minute % timeframe)
        seconds_to_next_nth_minute = minutes_to_next_nth_minute * 60 - now.second

        # Subtract 3 seconds (or however many seconds you need for self.warm_up_bot() to run)
        sleep_duration = max(seconds_to_next_nth_minute - self.bot_timeframe_delay_s, 0)

        # Sleep for the calculated duration
        time.sleep(sleep_duration)

        self.start_jobs()



    def start_jobs(self,):
        try:
            if self.replay:
                self.start_jobs_with_replay()
            else:
                self.start_jobs_live()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Exiting...")
            # send_msg_trade_update("KeyboardInterrupt detected. Exiting...")
            send_msg(message = {'text': f'{"="*30}\nTrading Bot FORCED shut down ... \n{"="*30}\n\n'})
        except Exception as e:
            logger.info(f"Exception occurred: {e}")
            traceback.print_exc()
            send_msg(message = {'text': f'{"="*30}\nTrading Bot CRASHED due to:\n{e}\n{"="*30}\n\n'})
    


    def start_jobs_with_replay(self,):
        # Turn off update so we dont want to get rate limited, and we dont even need updated data since we replaying!
        self.model_config["data_update"] = False
        # Load the historical data
        # this slices total data queried to specified replay window
        # This gets all available data till replay_window START

        df_to_replay, df_full_till_window_1 = self.load_historical_data(replay_i = None, live_data_safety_lock=False) 

        # Calculate the total number of events in the historical data
        # add this to exception output to see if it is slicing correctly
        # logger.info(f"\n\n{'='*10}\n\ndf_to_replay: \n{df_to_replay}\n\n{self.replay_window[0]}  --> {self.replay_window[1]}\ndf_to_replay type  --> {type(df_to_replay)}\n{'='*10}\n\n")
        # logger.info(f"\n\n{'='*10}\n\ndf_to_replay: \n[{self.replay_window[0]}:{self.replay_window[1]}\\n\n {self.replay_window[0]} --> {self.replay_window[1]}\n{'='*10}\n\n")
        total_timeframes = len(df_full_till_window_1[self.replay_window[0]:self.replay_window[1]])
        total_duration_in_m = total_timeframes * int(self.model_config["timeframe_to_trade"][:-1])/self.replay_speed
        
        # Calculate the duration of each event in seconds
        timeframe_duration_in_m = int(self.model_config["timeframe_to_trade"][:-1]) 
        timeframe_duration = timeframe_duration_in_m * 60

        # Calculate the simulated duration of each timeframe based on the replay_speed
        simulated_timeframe_duration = timeframe_duration / self.replay_speed

        # Iterate over the historical data
        # find index at which df_full_till_window_1 is self.replay_window[0]
        i_start = df_full_till_window_1.index.get_loc(self.replay_window[0])
        # i_start = min(self.model_config["memory_len"], total_timeframes)
        i_end = len(df_full_till_window_1[:self.replay_window[1]])
        # logger.info(f"\n{'='*100}\nREPLAY MODE - STARTING REPLAY\n{'='*100}\nETA: {total_duration_in_m} mins\n{self.model_config['memory_len']} -> {total_timeframes}\n\n")
        
        # total_mins_in_replay_window_int = (datetime.strptime(self.replay_window[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(self.replay_window[0], '%Y-%m-%d %H:%M:%S')).total_seconds()/60
        # est_time_completion = total_mins_in_replay_window_int/self.replay_speed
        # bot_start_title_msg = f"{'##'}{' '*10}REPLAY MODE{' '*10}{'##'}{' '*10} \nSTART ---> END: \n{self.replay_window[0]} ---> {self.replay_window[1]} \nEst time completion: {est_time_completion} minutes"
        
        
        for i in range(i_start, i_end, 1):
            # Run the bot cycle and bot update
            self.run_bot_cycle(replay_i=i)
            if (i*timeframe_duration_in_m) % self.bot_position_update_interval_m == 0:
                self.run_bot_update(replay_dt = str(df_full_till_window_1.index[i]))

            # Sleep for the simulated duration of the timeframe
            time.sleep(simulated_timeframe_duration)

        # Send a message to Slack when the replay is complete
        send_msg(message = {'text': f'\n\n{"="*40}\nREPLAY MODE - REPLAY COMPLETE\n{"="*40}\n\n'})

    def start_jobs_live(self,):
        # get timeframe data and get integer infront
        timeframe = int(self.model_config["timeframe_to_trade"][:-1])

        sched = BlockingScheduler()
        # wait for start of nth minute to start jobs:
        # Calculate the number of seconds until the next 5th minute
        now = datetime.now()
        minutes_to_next_nth_minute = timeframe - (now.minute % timeframe)
        seconds_to_next_nth_minute = minutes_to_next_nth_minute * 60 - now.second

        # Sleep until the next 5th minute
        # logger.info(f"Waiting for start of the next {timeframe}th minute to run the trading bot ... {seconds_to_next_nth_minute + self.bot_timeframe_delay_s} seconds")
        time.sleep(seconds_to_next_nth_minute + self.bot_timeframe_delay_s)
        
        dt_UTC, dt_SGT = get_dt_times_now()
        logger.info(f"---- Starting jobs AT {dt_UTC} ----")
        sched.add_job(self.run_bot_cycle, IntervalTrigger(minutes=timeframe))
    
        sched.add_job(self.run_bot_market_open_close_announcement, IntervalTrigger(minutes=1))  # Check every minute
        
        sched.add_job(self.run_bot_update, IntervalTrigger(minutes=self.bot_position_update_interval_m))
        # # Schedule the partial functions at specific times
        # schedule.every().minute.at(":00").do(self.run_bot_update, message='60 Minute Check')
        # schedule.every().minute.at(":15").do(self.run_bot_update, message='15 Minute Check')
        # schedule.every().minute.at(":30").do(self.run_bot_update, message='30 Minute Check')
        # schedule.every().minute.at(":45").do(self.run_bot_update, message='45 Minute Check')
        
        # TODO: ADD jobs for notification_manager and order_manager
        # sched.add_job(self.notification_manager)


        sched.start()



    def run_bot_market_open_close_announcement(self):
        now = datetime.now().time()
        if now == self.trading_start_time:
            self.send_msg(message = {'text':'Trading_bot wakes up'})
        elif now == self.trading_end_time:
            self.send_msg(message = {'text':'Trading_bot sleeps ...'})

    def warm_up_bot(self):
        # ============================================================================================================
        # 0) Initialise klines_manager AND signal_generator
        # ============================================================================================================

        self.klines_manager = load_ohlcv(window = self.model_config["data_window"],
                                            instruments_to_query = self.model_config["instruments_to_query"],
                                            instruments = self.model_config["instruments"],
                                            instrument_index_to_trade = self.model_config["instrument_index_to_trade"],  
                                            timeframes = self.model_config["timeframes_to_query"], 
                                            resample_to_list = self.model_config["resample_to_list"], 
                                            since = self.model_config["data_window"][0],  
                                            limits = self.model_config["limits"], 
                                            update = True,
                                            timeframe_to_trade = self.model_config["timeframe_to_trade"],
                                            memory_len = self.model_config["memory_len"], 
                                            verbose = False,
                                            tradable_times = self.model_config["clean_tradable_times"],
                                        )

        self.signal_generator = signal_generator(self.model_config)
                                                
        # logger.info(f"\n{'='*100}\n(0)(a) BOT WARMING UP --> Initial INITIAL model_state \n{'='*100}\n{self.model_state}\n{'='*100}\n\n")

        # ============================================================================================================
        # Warm up signal_generator
        # ============================================================================================================
        if self.replay:
            # if replay ENSURES live_data_safety_lock is False to allow for data slicing
            df_t0, df = self.load_historical_data(replay_i = None, live_data_safety_lock=False)
            self.df = df
            self.df_t0 = df_t0
        else:
            # if live trading ENSURES live_data_safety_lock is True to PREVENT data slicing
            self.df = self.load_historical_data(replay_i = None, live_data_safety_lock=True)
            self.df_t0 = self.df
            
        dfmtz = calc_sig().calc_signal(self.df_t0, self.model_config)

        model_state, df_backtested, df_trades, df_summary = self.signal_generator.get_trading_decision(dfmtz, self.model_state, self.model_config)
        self.model_state = model_state
        self.df_backtested = df_backtested
        self.df_trades = df_trades
        self.df_summary = df_summary

        tradable_times_str = " & ".join([f"{i[0]} to {i[-1]}" for i in self.model_config["tradable_times"]])
        for pos in ["L", "S"]:
            self.model_state[pos]["tradable_times"] = self.model_config["tradable_times"]
        # find row index i for df where rewind_time starts
        replay_dt = None
        if self.replay:
            replay_dt = datetime.strptime(self.replay_window[0], "%Y-%m-%d %H:%M:%S")
            # logger.info(f"\n\n{'='*10}\nreplay_dt: {replay_dt}\n{'='*10}\n\n")
            self.model_state = self.order_manager.update_model_state(self.model_state, self.model_config, order_type = self.order_type, replay_dt = replay_dt)
            # calculate number rows then scale up by time replay_speed to get measure of time 
            total_timeframes = len(self.df[self.replay_window[0]:self.replay_window[1]])

            i_start = self.df.index.get_loc(self.replay_window[0])
            i_end = len(self.df[:self.replay_window[1]])
            total_mins_in_replay_window_int = (datetime.strptime(self.replay_window[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(self.replay_window[0], '%Y-%m-%d %H:%M:%S')).total_seconds()/60
            est_time_completion = total_mins_in_replay_window_int/self.replay_speed
            bot_start_title_msg = f"\n\n{'='*40}\nREPLAY MODE\n{'='*40}\n\n{' '*10} \ni_start ({i_start}) ---> i_end ({i_end}): \n{self.replay_window[0]} ---> {self.replay_window[1]} {self.replay_speed} X SPEED \nEst time completion: {est_time_completion} minutes\n\n"
            t0 = self.replay_window[0]

            # logger.info(f"\n{'='*100}\n(0)(b) BOT WARMED UP --> model_state \n{'='*100}\n{self.model_state}\n{'='*100}\n\n")
        
        elif self.model_config["run_mode"] == "paper":
            self.model_state = self.order_manager.update_model_state(self.model_state, model_config = self.model_config, replay_dt = None)
            dt_UTC_now, _ = get_dt_times_now()
            bot_start_title_msg = f"trading_bot init - run_mode: paper"
            t0 = datetime.now().astimezone(pytz.UTC)

        elif self.model_config["run_mode"] == "live":
            self.model_state = self.order_manager.update_model_state(self.model_state, model_config = self.model_config, replay_dt = None)
            dt_UTC_now, _ = get_dt_times_now()
            bot_start_title_msg = f"trading_bot init - run_mode: live"
            t0 = datetime.now().astimezone(pytz.UTC)
        else:
            raise Exception(f"run_mode: {self.model_config['run_mode']} not recognised -> choose from: ['paper', 'live'] (replay counts as paper)")
        
        if not self.graceful_resume:
            send_msg(message = {'text': f'{bot_start_title_msg} - MODEL: {self.model_config["model_name"]}_{self.model_config["timeframe_to_trade"]} - Tradable times: {tradable_times_str}'})
        else:
            send_msg(message = {'text': f'{bot_start_title_msg} - MODEL: {self.model_config["model_name"]}_{self.model_config["timeframe_to_trade"]} - radable times: {tradable_times_str}'})
        # if plots enabled: 
        if self.prepare_plots_flag:
            self.prepare_plots(replay_dt=replay_dt)
        if self.publish_html_plotly_dashboard:
            html_file_path = f"{self.html_file_path}{self.html_file_name}.html"
        else:
            html_file_path = None
        send_msg_position_update(self.model_state, html_file_path=html_file_path)
        
        # logger.info(f"\n{'='*100}\n(0)(b) BOT WARMED UP --> model_state \n{'='*100}\n{self.model_state}\n{'='*100}\n\n")


                
  

    def load_order_manager(self, model_config, save_to_db_flag):
        if model_config["run_mode"] == "paper":
            logger.warning(f"\n\n{'='*10}\n\nPAPER TRADING MODE\n\n{'='*10}\n\n")
            order_manager  = Paper_Order_Manager(self.model_config, save_to_db_flag)
            self.order_type = "market"
        elif model_config["run_mode"] == "live":
            logger.warning(f"\n\n{'='*10}\n\nLIVE TRADING MODE\n\n{'='*10}\n\n")
            order_manager = KGI_Client(self.model_config)
            self.order_type = "market"
        else:
            raise Exception(f"run_mode: {model_config['run_mode']} not recognised")
        return order_manager



    def _run_bot_cycle(self, replay_i = None):
        t_now = datetime.now().astimezone(pytz.UTC)
        # =====================================================================
        # 1) Check model state ie; check for order fills
        # =====================================================================
        t_s_cycle = time.time()
        
        logger.info(f"\n{'='*100}\n(1) BOT CYCLE START: {t_now} \n{'='*100}\n\n")
        logger.info(f"\n{'='*100}\n(1)(a) Initial model_state \n{'='*100}\n{self.model_state}\n\n")
        
        # =====================================================================
        # 2) Update klines 
        # =====================================================================
        logger.info(f"\n{'='*100}\n(2) Update klines and signals \n{'='*100}\n\n")
        #TODO: Have to assert replay_i is None if live trading
 


        # Check if any pending orders, if so cancel but only if theres any new signal
        # Check if pending orders have been filled before trading
        if self.replay:
            # ENSURES replay_i is not None if replay
            df = self.load_historical_data(replay_i=replay_i, live_data_safety_lock=False)
            replay_dt = str(df.index[-1])
            # logger.info(f"\n\n{'='*10}\nreplay_dt: {replay_dt}\n{'='*10}\n\n")
            self.model_state = self.order_manager.update_model_state(self.model_state, model_config = self.model_config, order_type = self.order_type, replay_dt = replay_dt)
        else:
            # ENSURES replay_i is None if live trading
            df = self.load_historical_data(replay_i=None, live_data_safety_lock=True)
            self.model_state = self.order_manager.update_model_state(self.model_state, model_config = self.model_config, replay_dt = None)

        # logger.info(f"\n{'='*100}\n(2)(a) update_model_state \nmodel_state: \n{'='*100}\n{self.model_state}\n\n")
        # =====================================================================
        # 3) Analyse trade signal
        # =====================================================================
        # logger.info(f"\n{'='*100}\n(3) Analyse trade signal \n{'='*100}\n\n")

        dfmtz = calc_sig().calc_signal(df, self.model_config)
        model_state, df_backtested, df_trades, df_summary = self.signal_generator.get_trading_decision(dfmtz, self.model_state, self.model_config)
        self.model_state = model_state
        self.df_backtested = df_backtested
        self.df_trades = df_trades
        self.df_summary = df_summary

        logger.info(f"\n{'='*100}\n(3)(a) get_trading_decision DONE \nmodel_state: \n{'='*100}\n{self.model_state}\n\n")
        # =====================================================================
        # 4) Send orders
        # =====================================================================
        # This should all be done by another thread or handled by another cron job

        logger.info(f"\n{'='*100}\n(4) Orders \n{'='*100}\n\n")
        if self.publish_html_plotly_dashboard:
            df_trades = self.order_manager.consolidate_pnl(model_config = self.model_config, transactions_df= self.df_trades)

            self.prepare_plots(replay_dt=replay_dt)
            
            # html_file_path = f"{html_file_path}{self.model_config['instrument_to_trade']}_{self.model_config['model_name']}_{self.model_config['timeframe_to_trade']}"
            send_msg_trade_update(self.model_state, html_file_path=self.html_file_path)
        else:
            send_msg_trade_update(self.model_state, html_file_path=None)
        
        self.model_state = self.analyze_model_state_and_send_orders(self.model_state)
        # TODO: model_state_to_log = {"L": {"entry_now" model_state["L"]["entry_now"]}, } since too much info. keep only state changes
        logger.info(f"\n{'='*100}\n(4)(b) analyze_model_state_and_send_orders \nmodel_state: \n{'='*100}\n{self.model_state}\n\n")
       
        # =====================================================================
        # 5) Finish tasks
        # =====================================================================
        t_e_cycle = np.round(time.time()-t_s_cycle,3)
        logger.info(f"\n{'='*100}\n(5) BOT CYCLE END: ({t_e_cycle} s)\n{'='*100}\n\n")

        
        pickle_this(data=self.model_state, pickle_name=f"trading_bot_{self.model_config['model_name']}_{self.model_config['timeframe_to_trade']}", path="./database/model_states/")
                    
    def load_historical_data(self, 
                             replay_i = None, 
                             live_data_safety_lock = True, 
                             get_latest_data_only = False):

        if live_data_safety_lock:
            df = self.klines_manager.load_klines(stage=False,  
                            return_dict = False,
                            use_alt_volume = self.model_config["use_alt_volume"],
                            clean_base_data_flag = self.model_config["clean_base_data_flag"],
                            update=self.model_config["data_update"],
                            memory_len_lock = True)
            # return full df if live trading (all slicing for memory len done in load_Klines)
            return df
        else:

            if self.replay and replay_i is None:
                # LOAD FULL DATA FOR REPLAY to get est time completion then slice  FOR WARMUP
                df = self.klines_manager.load_klines(stage=False,  
                            return_dict = False,
                            use_alt_volume = self.model_config["use_alt_volume"],
                            clean_base_data_flag = self.model_config["clean_base_data_flag"],
                            update=self.model_config["data_update"],
                            memory_len_lock = False)
                        

                # df = df[(df.index >= self.replay_window[0])]
                # df = df[(df.index <= self.replay_window[1])].copy()
                df_t0 = df[(df.index <= self.replay_window[0])].copy()
                return df_t0, df
            if self.replay and replay_i is not None:
                # LOAD FULL DATA then slice for replay
                df = self.klines_manager.load_klines(stage=False,  
                            return_dict = False,
                            use_alt_volume = self.model_config["use_alt_volume"],
                            clean_base_data_flag = self.model_config["clean_base_data_flag"],
                            update=self.model_config["data_update"],
                            memory_len_lock = False)
                # slice the data as if live trading
                df = df.iloc[replay_i-self.model_config["memory_len"]:replay_i]
                return df

    def analyze_model_state_and_send_orders(self, model_state):
        # this condition can be streamlined but for now it is left as is for clarit
        # TODO: THIS IS NOT BEING TRIGGERED?! issue is in signal_geenrator get_trading_decision - why is it not reflecting exit now or entry now?
        if model_state["L"]["entry_now"] or model_state["L"]["exit_now"] or model_state["S"]["entry_now"] or model_state["S"]["exit_now"]:

            # logger.info(f"\n{'='*100}\n(4)(a) analyze_model_state_and_send_orders: SIGNAL TRIGGERED \nmodel_state: \n{'='*100}\n{model_state}\n\n")
            model_state = self.order_manager.send_orders(model_state)
        else: 
            logger.info(f"\n{'='*100}\n(4)(a) analyze_model_state_and_send_orders: NONE \nmodel_state: \n{'='*100}\n{model_state}\n\n")
        
        return model_state



    def run_bot_cycle(self, replay_i = None):
        try:
            now = datetime.utcnow().time()

            if not self.trading_start_time <= now <= self.trading_end_time:
                logger.info(f'OUTSIDE TRADING HOURS - time now: {now} not within trading times: {self.trading_start_time} --> {self.trading_end_time}')
                return

            self._run_bot_cycle(replay_i = replay_i)
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt detected. Exiting...")
            # send_msg_trade_update("KeyboardInterrupt detected. Exiting...")
            send_msg(message = {'text': f'{"="*30}\nTrading Bot FORCED shut down ... \n{"="*30}\n\n'})
            
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            traceback.print_exc()
            send_msg(message = {'text': f'{"="*30}\nTrading Bot ERROR occurred due to:\n{e}\n{"="*30}\n\n'})
        


    def run_bot_update(self, replay_dt = None):
        if self.replay:
            self.model_state = self.order_manager.update_model_state(self.model_state, model_config = self.model_config, order_type = self.order_type, replay_dt=replay_dt)
        else:
            now = datetime.utcnow().time()
            if not self.trading_start_time <= now <= self.trading_end_time:
                logger.info(f'OUTSIDE TRADING HOURS - time now: {now} not within trading times: {self.trading_start_time} --> {self.trading_end_time}')
                return
            else:
                self.model_state = self.order_manager.update_model_state(self.model_state,  model_config = self.model_config, replay_dt=None)
        
        # if plots enabled: 
        if self.prepare_plots_flag:
            self.prepare_plots(replay_dt=replay_dt)
        if self.publish_html_plotly_dashboard:            
            # html_file_path = f"{html_file_path}{self.model_config['instrument_to_trade']}_{self.model_config['model_name']}_{self.model_config['timeframe_to_trade']}"
            send_msg_position_update(self.model_state, html_file_path=self.html_file_path)
        else:
            send_msg_position_update(self.model_state, html_file_path=None)






    def prepare_plots(self,replay_dt = None):
        df_trades = self.order_manager.consolidate_pnl(model_config = self.model_config, transactions_df = self.df_trades)
        if self.replay:
            plots(model_config = self.model_config,
                    df_backtested = self.df_backtested,
                    df_trades = self.df_trades,
                    df_summary = self.df_summary ,
                    replay_dt = replay_dt,
                    html_file_path = self.html_file_path,
                    publish = True,
                    convert_tz = None,
                    cols_to_plot= self.cols_to_plot,
                    subplot_titles= self.subplot_titles,
                    row_heights = self.row_heights,
                    verbose=  False,
                    width= 2000,
                    height= 1600,manual_tail_window=None)
        else:
            plots(model_config = self.model_config,
                    df_backtested = self.df_backtested,
                    df_trades = self.df_trades,
                    df_summary = self.df_summary ,
                    replay_dt = None,
                    html_file_path = self.html_file_path,
                    publish = True,
                    convert_tz = None,
                    cols_to_plot= self.cols_to_plot,
                    subplot_titles= self.subplot_titles,
                    row_heights = self.row_heights,
                    verbose=  False,
                    width= 2000,
                    height= 1600,manual_tail_window=None)
            
        # Check if can open file
        try:
            with open(f"{self.html_file_path}{self.html_file_name}.html", 'rb') as file:
                # Do something with the file
                pass
        except IOError as e:
            raise Exception(f"Could not open file: {self.html_file_path}{self.html_file_name}.html\n{e}")
            
        