import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from importlib import import_module

from performance_analytics.metrics import backtest_summary
from performance_analytics.backtest_plot import backtest_plots_ppt_SR, backtest_plots_simplified, backtest_plots_ppt
import models


def _backtest(model_name,
              df0: pd.DataFrame,
              kline_to_trade = "5m_close",
              klines_tradable = "tradable",
              klines_session_closing = "session_closing",
              volume_to_trade = "5m_volume",
              fee=0.001,
              slippage = 0.0003, # 3bps for slippage
              long_notional=1000, # to be changed to dynamic position sizing
              short_notional=1000,
              position_sizing_to_trade = None,
              signals = ["sig", "L_uq", "L_lq", "S_uq", "S_lq"], 
              sig_lag=0,
              signal_function = None,
              window=None,
              min_holding_period=1,
              max_holding_period=10000,
              max_positions = 5,
              trail_SL = -0.004,
              trail_TP = 0.0002,
              trail_increment = 0.0001,
              disable_tqdm= True,
              reduce_only = False,
              **kwargs,
              ):
    if window is None:
        df=df0.copy()
    else:
        df=df0.copy()[window[0]:window[1]]
        
    # signals related
    signals_dict = {}

    for signal in signals:
        # print(signal)
        signals_dict[signal] = df[signal].values

        
    # print(f"kwargs from _backtest: {kwargs}")
    # =============================================================================
    # SELECT SIGNAL FUNCTION
    # =============================================================================
    model = import_module(f"models.{model_name}")
    # print(model)
    if (signal_function is None) or (signal_function == "label"):
        _get_signal = model.get_signal
    elif signal_function == "qtl":
        _get_signal = model.get_signal_qtl
    elif signal_function == "default":
        _get_signal = model.get_signal_default
    elif signal_function == "tide":
        _get_signal = model.get_tide_sig
    elif signal_function == "tide_TP":
        _get_signal = model.get_tide_sig_TP
    elif signal_function == "z_sig":
        _get_signal = model.get_z_sig
    elif signal_function == "z_sig_TP":
        _get_signal = model.get_z_sig_TP
    else:
        raise Exception(f"{signal_function} ---> is not valid signal function")
        
    # Piggyback signals dict for strat type TF / MR
    signals_dict["L"] = None         
    signals_dict["S"] = None  
    # positions
    np_long_positions = np.full(len(df), np.nan)
    np_short_positions = np.full(len(df), np.nan)
    
    
    # px/qty/cost details
    np_long_id = np.full(len(df), np.nan)
    np_long_entry = np.full(len(df), np.nan)
    np_long_cost = np.full(len(df), np.nan)
    np_long_qty = np.full(len(df), np.nan)
    np_long_exit = np.full(len(df), np.nan)

    
    np_short_id = np.full(len(df), np.nan)
    np_short_entry = np.full(len(df), np.nan)
    np_short_cost = np.full(len(df), np.nan)
    np_short_qty = np.full(len(df), np.nan)
    np_short_exit = np.full(len(df), np.nan)

        
    # pnl details
    np_pnl = np.full(len(df), np.nan)
    
    np_long_rpnl = np.full(len(df), np.nan)
    np_long_pnl = np.full(len(df), np.nan) # this is used for unrealised pnl
    np_long_pnl_pct = np.full(len(df), np.nan) # this is for trailing stop
    np_long_fees = np.full(len(df), np.nan)
    
    np_short_rpnl = np.full(len(df), np.nan)
    np_short_pnl = np.full(len(df), np.nan) # this is used for unrealised pnl
    np_short_pnl_pct = np.full(len(df), np.nan) # This is for trailing stop
    np_short_fees = np.full(len(df), np.nan)
    
    # last price
    np_closePx = df[kline_to_trade].values
    np_vol = df[volume_to_trade].values
    np_tradable = df[klines_tradable].values
    np_session_closing = df[klines_session_closing].values
    if position_sizing_to_trade is None:
        np_position_sizing = np.full(len(df), 1)
    else:
        np_position_sizing = df[position_sizing_to_trade].values
    
    # initialize state flags
    in_long_position  = False
    in_short_position = False
    long_id = 1
    short_id = 1
    

    long_closeIdx = 0
    short_closeIdx = 0
    # t0=time.time()
    for i in tqdm(range(len(df)), disable=disable_tqdm):
        # TRADING TIMES FORMAT:  [02:45","05:25"] and ["07:15","09:50"]
        
        
        # Should be exact times
        tradable = np_tradable[i]
        session_closing = np_session_closing[i]
        
        # =============================================================================
        # ENTRIES
        # =============================================================================

        # ---------- #
        # ENTER LONG
        # ---------- #
        if reduce_only:
            long_position_available = not in_long_position and not in_short_position
            short_position_available = not in_short_position and not in_long_position
        else:
            long_position_available = not in_long_position
            short_position_available = not in_short_position

        if long_position_available and tradable and not session_closing and long_notional>0:
            signal = _get_signal(i,np_closePx,signals_dict, sig_lag=sig_lag, position="long",side="buy",long_closeIdx = long_closeIdx, short_closeIdx = short_closeIdx, **kwargs)
            if signal:
                # Trackers
                np_long_positions[i] = 1
                np_long_id[i] = long_id
                in_long_position = True
                long_openIdx = i
                
                # px/qty/cost details
                long_entry_Px =  np_closePx[i]*(1+slippage) 
                np_long_entry[i] = long_entry_Px
                long_cost = long_notional * np_position_sizing[i]
                np_long_cost[i] = long_cost
                np_long_qty[i] = long_cost/long_entry_Px
                signals_dict["L"] = long_entry_Px
                
        # ---------- #
        # ENTER SHORT
        # ---------- #
        if short_position_available and tradable and not session_closing and short_notional>0:
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=sig_lag, position="short",side="buy",short_closeIdx = short_closeIdx, long_closeIdx = long_closeIdx, **kwargs)
            if signal:
                # Trackers
                np_short_positions[i] = -1
                np_short_id[i] = short_id
                in_short_position = True
                short_openIdx = i
                
                # px/qty/cost details
                short_entry_Px = np_closePx[i]*(1-slippage) 
                np_short_entry[i] = short_entry_Px
                short_cost = short_notional * np_position_sizing[i]
                np_short_cost[i] = short_cost
                np_short_qty[i] = short_cost/short_entry_Px
                signals_dict["S"] = short_entry_Px
                
        # =============================================================================
        # EXITS
        # =============================================================================     
        
        # ========== #
        # LONG
        # ========== #
        if in_long_position and (i > long_openIdx): 
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=sig_lag, position="long",side="sell",long_openIdx = long_openIdx, **kwargs)
            
            # ---------- #
            # EXIT LONG
            # ---------- #   
            min_holding_period_flag = i >= (long_openIdx + min_holding_period)
            signal_triggered_flag = (signal and tradable and not session_closing)
            tradable_but_session_closing =  (tradable and session_closing)
            max_holding_period_flag =  (i >= (long_openIdx + max_holding_period))

            if min_holding_period_flag and ( signal_triggered_flag or tradable_but_session_closing) or max_holding_period_flag: 
                # Trackers
                long_closeIdx = i
                np_long_positions[i] = 0
                in_long_position = False

                # px/qty/cost details
                long_exit_Px = np_closePx[i]*(1-slippage) 
                np_long_exit[i] = long_exit_Px
                np_long_qty[i] = np_long_qty[i-1]
                np_long_cost[i] = long_exit_Px * np_long_qty[i]
                
                # pnl details
                discrete_Long_pnl = (long_exit_Px-long_entry_Px)*np_long_qty[i]
                fees = np_long_qty[i]*long_exit_Px*fee + np_long_qty[i]*long_entry_Px*fee
                discrete_Long_pnl-=fees
                
                # Records
                np_long_fees[i] = fees
                np_long_pnl[i] = 0 # take pnl from realised instead 
                np_pnl[i] = discrete_Long_pnl 
                np_long_pnl_pct[i] = (long_exit_Px-long_entry_Px)/long_entry_Px
                np_long_rpnl[i] = discrete_Long_pnl
                np_long_id[i] = long_id
                long_id += 1
                signals_dict["L"] = None
                
            # elif i <= ((long_openIdx + min_holding_period)
                
            # ---------- #
            # STAY LONG
            # ---------- #
            else:
                np_long_id[i] = long_id
                np_long_positions[i] = np_long_positions[i-1]      
                np_long_cost[i] = np_long_cost[i-1]
                np_long_qty[i] = np_long_qty[i-1]
                np_long_pnl[i] = (np_closePx[i]-long_entry_Px)*np_long_qty[i]
                np_long_pnl_pct[i] = (np_closePx[i]-long_entry_Px)/long_entry_Px
            
                
        # ========== #
        # SHORT
        # ========== #   
        if in_short_position and (i > short_openIdx):
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=sig_lag, position="short",side="sell",short_openIdx = short_openIdx, **kwargs)
            
            # ---------- #
            # EXIT SHORT
            # ---------- #
            min_holding_period_flag = i >= (short_openIdx + min_holding_period) 
            signal_triggered_flag = (signal and tradable and not session_closing)
            tradable_but_session_closing = (tradable and session_closing)
            max_holding_period_flag = (i >= (short_openIdx + max_holding_period))                
                
            if min_holding_period_flag and (signal_triggered_flag or tradable_but_session_closing) or max_holding_period_flag:
                # Trackers
                short_closeIdx = i
                np_short_positions[i] = 0
                in_short_position = False

                # px/qty/cost details
                short_exit_Px = np_closePx[i]*(1+slippage)
                np_short_exit[i] = short_exit_Px
                np_short_qty[i] = np_short_qty[i-1]
                np_short_cost[i] = short_exit_Px * np_short_qty[i]
                
                # pnl details
                discrete_Short_pnl = (short_entry_Px-short_exit_Px)*np_short_qty[i]
                fees = np_short_qty[i]*short_exit_Px*fee + np_short_qty[i]*short_entry_Px*fee
                discrete_Short_pnl-=fees
                
                # Records
                np_short_fees[i] = fees
                np_short_pnl[i] = 0 #discrete_Short_pnl
                np_pnl[i] = discrete_Short_pnl
                np_short_rpnl[i] = discrete_Short_pnl
                np_short_pnl_pct[i] = (short_entry_Px-short_exit_Px)/short_exit_Px
                np_short_id[i] = short_id
                short_id += 1
                signals_dict["S"] = None
                
            # ---------- #
            # STAY SHORT
            # ---------- #
            else:
                np_short_id[i] = short_id
                np_short_positions[i] = np_short_positions[i-1]
                np_short_cost[i] = np_short_cost[i-1]
                np_short_qty[i] = np_short_qty[i-1]
                np_short_pnl[i] = (short_entry_Px-np_closePx[i])*np_short_qty[i]
                np_short_pnl_pct[i] = (short_entry_Px-np_closePx[i])/np_closePx[i]
                
                

                    
    # END BACKTEST, collate
    
    
    df["L_id"] = np_long_id
    df["L_positions"] = np_long_positions
    df["L_entry_price"] = np_long_entry
    df["L_cost"] = np_long_cost
    df["L_qty"] = np_long_qty
    df["L_exit_price"] = np_long_exit
    df["L_fees"] = np_long_fees
    df["L_pnl"] = np_long_pnl
    df["L_pnl%"] = np_long_pnl_pct
    df["L_rpnl"] = np_long_rpnl
    
    df["S_id"] = np_short_id
    df["S_positions"] = np_short_positions
    df["S_entry_price"] = np_short_entry
    df["S_cost"] = np_short_cost
    df["S_qty"] = np_short_qty
    df["S_exit_price"] = np_short_exit
    df["S_fees"] = np_short_fees
    df["S_pnl"] = np_short_pnl
    df["S_pnl%"] = np_short_pnl_pct
    df["S_rpnl"] = np_short_rpnl

    
    df["A_rpnl"] = np_pnl
    df["B_pnl"] = df[kline_to_trade].pct_change()
    
    return df


def backtest(model_name, 
             df0,
             timeframe,
              kline_to_trade = "5m_close",
              klines_tradable = "tradable",
              volume_to_trade = "5m_volume",
              tradable_times = [('02:45', '05:25'),('07:15', '09:50')],
              days_of_the_week_to_trade = [0,1,2,3,4], # 0 is Monday, 6 is Sunday
            #   closing_session_times = [('05:20', '05:30'),('09:40', '09:50')],
              position_sizing_to_trade = None,
              fee=0.0007,
              slippage = 0.0003, # 1bps for slippage
              long_equity = 50000,
              short_equity = 50000,
              long_notional=10000,
              short_notional=10000,
              signals = ["sig", "L_uq", "L_lq", "S_uq", "S_lq"], 
              sig_lag=0,
              signal_function=None,
              min_holding_period=1,
              max_holding_period=1e6,
              max_positions = 5,
              plots=True,
              produce_signal =False,
              signal_name = None,
              window=None,
              horizon_labels=None,
              show_B=True,
              show_LS=False,
              figsize=(16,7),
              file_name="model",
              title="test",
              metrics_table_item_to_drop= ["Lose Rate %"],
              diagnostics_verbose=False,
              trail_SL =  None, #-0.0004,
              trail_TP = None, #0.0002,
              trail_increment = None,
              N = None,
              reduce_only=False,
              show_rolling_SR = False,
              **kwargs,
              ): #0.0001):
    
    # if produce_signal:
    #     _window = window
    # else:
    #     _window = window
    
    t0 = time.time()
    df = df0.copy()
    # if N is None:
    #     print("calculating N...")
    #     Ndf = df0.copy()
    #     Ndf["datetime"]=Ndf.index
    #     N = Ndf.groupby(Ndf["datetime"].dt.year).count().max().max()
    dur_N = np.round(time.time()-t0,3)
    # =============================================================================
    # TRADING TIMES AND SESSION CLOSING FLAGS
    # =============================================================================
    # Convert days of the week to trade into a set for faster lookup
    t0 = time.time()
    days_of_the_week_to_trade = set(days_of_the_week_to_trade)
    if ("tradable" in df.columns) and ("session_closing" in df.columns):
        pass # already calculated
    elif (tradable_times is None):
        df["tradable"] = True
        df["session_closing"] = False
    else:
        
        trading_times_index = []
        # session_closing_index = []
        # for tradable_time, session_closing_time in zip(tradable_times, closing_session_times):
        for tradable_time in tradable_times:
            df_filtered = df[df.index.dayofweek.isin(days_of_the_week_to_trade)]
            # trading_times_index += list(df.between_time(tradable_time[0], tradable_time[1]).index)
            trading_times_index += list(df_filtered.between_time(tradable_time[0], tradable_time[1]).index)
            # session_closing_index += list(df.between_time(session_closing_time[0], session_closing_time[1]).index)
        
        # Add flags to main dataframe
        df["tradable"] = df.index.isin(trading_times_index)
        df["tradable"] = df["tradable"].fillna(False).astype(bool)
        # print(~df["tradable"].shift(-1).tail(24))
        # df["session_closing"] = df.index.isin(session_closing_index)
        df['session_closing'] = df['tradable'] & (~df['tradable']).shift(-1)
        df['session_closing'] = df['session_closing'].fillna(False)
    dur_trading_times = np.round(time.time()-t0,2)
    
    # =============================================================================
    # START BACKTEST
    # =============================================================================
    if plots is False:
        disable_tqdm = True
    else:
        disable_tqdm = False
    t0=time.time()
    df = _backtest(model_name, 
                   df,
                   kline_to_trade = kline_to_trade,
                   klines_tradable = "tradable",
                   volume_to_trade = volume_to_trade,
                   position_sizing_to_trade=position_sizing_to_trade,
                   fee = fee,
                   slippage = slippage, # 1bps for slippage
                   long_notional = long_notional,
                   short_notional = short_notional,
                   signals = signals, 
                   sig_lag=sig_lag,
                   signal_function=signal_function,
                   window = window,
                   min_holding_period = min_holding_period,
                   max_holding_period=max_holding_period,
                   max_positions = max_positions,
                   trail_SL = trail_SL,
                   trail_TP = trail_TP,
                   trail_increment = trail_increment,
                   disable_tqdm=disable_tqdm,
                   reduce_only=reduce_only,
                   **kwargs)
    dur_backtest = np.round(time.time()-t0,2)
    # Add tradable column to show when to trade
    
    
    # =============================================================================
    # Output full backtest results
    # =============================================================================
    if produce_signal is False:
        t0=time.time()
        df_backtested, df_trades, df_summary = backtest_summary(df,
                                                                long_equity,
                                                                short_equity, 
                                                                fee=fee*10000,
                                                                timeframe=timeframe,
                                                                show_rolling_SR=show_rolling_SR,
                                                                )
        if diagnostics_verbose: print(df_trades)
        if len(df_trades) == 0:
            # raise Exception("No trades generated")
            if diagnostics_verbose: print(f"\n\n{'='*30}\nNo trades generated\n{'='*30}\n\n")
            return  df_backtested, df_trades, df_summary
        dur_metrics = np.round(time.time()-t0,2)
        # print(df_backtested.columns)
        t0=time.time()
        if plots == "simple":
            backtest_plots_simplified(df_backtested,
                                      df_summary[["total", "longs only", "shorts only"]], 
                                      horizon_labels=horizon_labels,
                                      show_B = show_B, 
                                      title=title,
                                      figsize=figsize,
                                      fees = fee,
                                      kline_to_trade = kline_to_trade)
        elif plots is True:
            backtest_plots_ppt(df_backtested,
                               df_trades,
                                df_summary,#[["total","buyhold"]], 
                                horizon_labels=horizon_labels,
                                show_B = show_B, 
                                show_LS= show_LS,
                                title=title,
                                figsize=figsize,
                                fees = fee,
                                kline_to_trade = kline_to_trade,
                                to_drop=metrics_table_item_to_drop,
                                file_name = file_name,
                                show_rolling_sr=show_rolling_SR)
        elif plots == "SR":
            backtest_plots_ppt_SR(df_backtested,
                               df_trades,
                                df_summary,#[["total","buyhold"]], 
                                horizon_labels=horizon_labels,
                                show_B = show_B, 
                                show_LS= show_LS,
                                title=title,
                                figsize=figsize,
                                fees = fee,
                                kline_to_trade = kline_to_trade,
                                to_drop=metrics_table_item_to_drop,
                                file_name = file_name,
                                show_rolling_sr=show_rolling_SR)
        elif plots is False:
            pass
        
        dur_plots = np.round(time.time()-t0,2)
            
        # print(df_summary)
        if diagnostics_verbose: print(f"\nBacktesting {df_backtested.index[0]} to {df_backtested.index[-1]} \n ({len(df)} rows)\nRuntimes\ndur_N: {dur_N} s\ndur_trading_times: {dur_trading_times} s\nbacktesting: {dur_backtest} s\nmetrics calc: {dur_metrics} s\nplots calc: {dur_plots} s")
        return df_backtested,df_trades,df_summary
    
    
    # =============================================================================
    # Output signals only
    # =============================================================================
    else:
        if diagnostics_verbose: print(f"Runtime:\nGenerating signals {len(df)} rows: {dur_backtest}s")
        # if signal_name is None:
        #     df0["L_signal"] = df["L_positions"]
        #     df0["L_signal"].fillna(0,inplace=True)
        # else:
        #     df0[signal_name] = df["L_positions"]
        #     df0[signal_name].fillna(0,inplace=True)
        if signal_name is None:
            # df["S_positions"] = -1 * df["S_positions"]
            df0["signal"] = df[["L_positions","S_positions"]].sum(axis=1,skipna=True)
            df0["signal"].fillna(0,inplace=True)
        else:
            # df["S_positions"] = -1 * df["S_positions"]
            df0[signal_name] = df[["L_positions","S_positions"]].sum(axis=1,skipna=True)
            df0[signal_name].fillna(0,inplace=True)
        return df0
#%%

if __name__ == "__main__":
    from utils import pickle_helper

    df = pickle_helper.pickle_this(pickle_name = "df_backtest_debugging", path = "./backtesters/debug/")
# %%
