import pandas as pd
import numpy as np
import time
from signal_managers import rolling_sr


def calc_position_drawdown(continuous_ret): # equivalent to returns.cumsum() #
    continuous_ret=1+continuous_ret
    prev_peak = continuous_ret.cummax()
    dd = (continuous_ret-prev_peak)/prev_peak 
    return dd

# def calc_equity_drawdown(equity):
#    cumulative_returns = equity.pct_change().cumsum()
#    running_max = cumulative_returns.cummax()
#    drawdown = ((cumulative_returns-running_max)/running_max) #*100
#    drawdown = drawdown.fillna(0)
   

#    return drawdown

def calc_equity_drawdown(equity):
    """
    Calculate drawdown by consuming equity
    """
    cumret = equity.pct_change(1).cumsum()

    dd= (cumret- cumret.cummax())*100

    return dd


def backtest_summary(df0,long_starting_equity,short_starting_equity, N=255*24*12, rolling_sr_window=None, fee=1, min_bar_size=5, timeframe="3h"): # sr=12*24*20*3
    df=df0.copy()
    total_equity = long_starting_equity + short_starting_equity
    summary_index = ["L","S","A","B"]
    # print(df.columns)
    initial_column_names = list(df.columns)
    
    df["A_rpnl"] = df[["L_rpnl","S_rpnl"]].sum(axis=1, min_count=1)
    
    # CUM PNL CALCULATIONS
    df["cum_L_rpnl"] = df['L_rpnl'].cumsum().fillna(method="ffill")
    df["cum_S_rpnl"] = df['S_rpnl'].cumsum().fillna(method="ffill")
    df["cum_A_rpnl"] = df[["cum_L_rpnl","cum_S_rpnl"]].sum(axis=1)
    # this will inflate cum pnl%!! so have to 
    df["L_pnl"].fillna(method="ffill", inplace=True)
    df["S_pnl"].fillna(method="ffill", inplace=True)
    
    df["cum_L_pnl"]=df[["cum_L_rpnl","L_pnl"]].sum(axis=1)
    df["cum_S_pnl"]=df[["cum_S_rpnl","S_pnl"]].sum(axis=1)
    df["cum_A_pnl"]=df[["cum_L_pnl","cum_S_pnl"]].sum(axis=1)
    
    df["cum_B_pnl"] = (((df["B_pnl"]+1).cumprod()-1)*total_equity)
    
    # % Equity Calculations
    df["cum_B_pnl%"] = (total_equity+df["cum_B_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_A_pnl%"] = (total_equity+df["cum_A_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_L_pnl%"] = (long_starting_equity+df["cum_L_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_S_pnl%"] = (short_starting_equity+df["cum_S_pnl"]).pct_change().add(1).cumprod()*100
    
    # Drawdown calculations
    df["dd_B"]=calc_equity_drawdown(df["cum_B_pnl"]+long_starting_equity+short_starting_equity)
    df["dd_A"]=calc_equity_drawdown(df["cum_A_pnl"]+long_starting_equity+short_starting_equity)
    df["dd_L"]=calc_equity_drawdown(df["cum_L_pnl"]+long_starting_equity)
    df["dd_S"]=calc_equity_drawdown(df["cum_S_pnl"]+short_starting_equity)
    
    
    # ROLLING SHARPE CALCULATION ---> THIS TAKES QUITE A LOT OF TIME TO PROCESS ~ 1minute
    # Consider numbarising this, it is possible
    # TODO: numbarise this
    t0=time.time()
    
    if rolling_sr_window is not None:
        print(f"rolling_sr_window is not None: {rolling_sr_window is not None}")
        try:
            df["A_rolling_SR"] = rolling_sr.calc_rolling_sr(df["cum_A_pnl%"].pct_change().values, window=rolling_sr_window)*np.sqrt(N)
            df["L_rolling_SR"] = rolling_sr.calc_rolling_sr(df["cum_L_pnl%"].pct_change().values, window=rolling_sr_window)*np.sqrt(N)
            df["S_rolling_SR"] = rolling_sr.calc_rolling_sr(df["cum_S_pnl%"].pct_change().values, window=rolling_sr_window)*np.sqrt(N)
        except Exception as e:
            print(f"ERROR: \n {e}")
    
    # max daily trades charts
    
    
    trade_cols = ['L_id',
                'L_positions',
                'L_entry_price',
                'L_cost',
                'L_qty',
                'L_exit_price',
                'L_pnl',
                'L_rpnl',
                'L_fees',
                'S_id',
                'S_positions',
                'S_entry_price',
                'S_cost',
                'S_qty',
                'S_exit_price',
                'S_pnl',
                'S_rpnl',
                'S_fees',
                'A_rpnl',
                'cum_B_pnl',
                'cum_A_pnl',
                'cum_L_pnl',
                'cum_L_rpnl',
                'cum_S_pnl',
                'cum_S_rpnl']

    # df1=df[initial_column_names+trade_cols]
    trades = df[trade_cols].dropna(subset=['L_entry_price','L_exit_price','S_entry_price','S_exit_price'],how="all")
    # print(df.columns)
    # =============================================================================
    # METRICS CALCULATIONS
    # =============================================================================
    
    # Number of trades
    number_of_trades = {}
    for c in summary_index:
        if c == "B":
            number_of_trades[c] = 1
        elif c == "A":
            number_of_trades[c] = df[['L_id','S_id']].max().sum()
        else:
            number_of_trades[c] = df[f'{c}_id'].max()
            
            
    # Time in trade
    time_in_trade_mean = {}
    time_in_trade_max = {}
    time_in_trade_min = {}
    total_hours = len(df)
    timeframe_int = int(timeframe[:-1])
    timeframe_interval = timeframe[-1:]
    for c in summary_index:
        if c == "B":
            time_in_trade_max[c] = total_hours*timeframe_int
            time_in_trade_min[c] = total_hours*timeframe_int
            time_in_trade_mean[c] = total_hours*timeframe_int
        elif c in ["S","L"]:
            time_in_trade_max[c] = df[f"{c}_id"].value_counts().max()*timeframe_int #max(len(c_trades_grouped.groups))
            time_in_trade_min[c] = df[f"{c}_id"].value_counts().min()*timeframe_int #min(len(c_trades_grouped.groups))
            time_in_trade_mean[c] = df[f"{c}_id"].value_counts().median()*timeframe_int
        elif c == "A":
            time_in_trade_max[c] = max(time_in_trade_max.values())
            time_in_trade_min[c] = min(time_in_trade_min.values())
            time_in_trade_mean[c] = np.mean([time_in_trade_mean["L"],time_in_trade_mean["S"]])

    

    # winrate
    winrate = {}
    loserate = {}
    drawrate = {}
    total_trades = {}
    for c in summary_index:
        if c == "B":
            winrate[c] = 0
            total_trades[c] = 1
        else:#if c in ["S","L"]:
            wins =  trades[trades[f'{c}_rpnl']>0][f'{c}_rpnl'].count()
            lose =  trades[trades[f'{c}_rpnl']<0][f'{c}_rpnl'].count()
            draw = trades[trades[f'{c}_rpnl']==0][f'{c}_rpnl'].count()
            
            # Use trade ids to calculate number of trades instead of win lose counts
            # total_trades[c] = wins+lose
            if c in ["S","L"]:
                try:
                    no_total_trades = trades[f'{c}_id'].dropna()[-1]
                except Exception as e:
                    no_total_trades = 0
                total_trades[c] = no_total_trades
                
                winrate[c] = wins/(total_trades[c])*100
                loserate[c] = lose/(total_trades[c])*100
                drawrate[c] = draw/(total_trades[c])*100
                
            elif c in ["A"]:
                total_trades[c] = total_trades["S"]  + total_trades["L"] 
                
                winrate[c] = wins/(total_trades[c])*100
                loserate[c] = lose/(total_trades[c])*100
                drawrate[c] = draw/(total_trades[c])*100
 
            
    
    # profit factor/best/worst trades
    p2g = {}
    pnl_median = {}
    pnl_mean = {}
    avg_wins = {}
    med_wins = {}
    
    avg_loss = {}
    med_loss = {}
    for c in summary_index:
        if c == "B":
            w_trades =  df[df[f'{c}_pnl']>0][f'{c}_pnl']
            l_trades = df[df[f'{c}_pnl']<0][f'{c}_pnl']
            
            gains =  0 #w_trades.sum()
            pains =  0# l_trades.sum()
            p2g[c] =  0 #gains/np.abs(pains)
            
            avg_wins[c] = 0 #w_trades.mean()
            med_wins[c] = 0 #w_trades.median()
            
            pnl_median[c] = 0
            pnl_mean[c] = 0
            
            avg_loss[c] = 0 #l_trades.mean()
            med_loss[c] = 0 #l_trades.median()
        else:
            w_trades =  df[df[f'{c}_rpnl']>0][f'{c}_rpnl']
            l_trades = df[df[f'{c}_rpnl']<0][f'{c}_rpnl']
            
            gains =  w_trades.sum()
            pains =  l_trades.sum()
            p2g[c] =  gains/np.abs(pains)
            
            avg_wins[c] = w_trades.mean()
            med_wins[c] = w_trades.median()
            
            pnl_median[c] = df[f'{c}_rpnl'].median()
            pnl_mean[c] = df[f'{c}_rpnl'].mean()
            
            avg_loss[c] = l_trades.mean()
            med_loss[c] = l_trades.median()
    # Sharpe
    # for strat in ["buyhold"]

    # sharpe = trades['long_rpnl'].mean() / trades['long_rpnl'].std() * np.sqrt(number_of_trades) 
    sharpes = {}

    
    for c in summary_index:
        if c == "B":
            ret = df[f"cum_{c}_pnl%"].diff()
            if ret.std() == 0:
                sharpes[c] = 0
            else:
                sharpes[c] = (ret.mean() / ret.std()) *np.sqrt(N)
        else:
            # Since hourly  #total_trades[c] / ((df.index[-1] - df.index[0]).days) # where N is number of trades in a day
            # ret = df[f"cum_{c}_pnl%"]
            # ret = df[f'{c}_rpnl']
            ret = df[f"cum_{c}_rpnl"].diff()
            
            if ret.std() == 0:
                sharpes[c] = 0
            else:
                sharpes[c] = (ret.mean() / ret.std()) *np.sqrt(N)
    
    # def sharpe(returns,N):
    # # >1 is good
    #     return returns.mean()*N/returns.std()/ np.sqrt(N)
    # def sortino(returns,N):
    #     # 0 - 1.0 is suboptimal, 1> good, 3> very good
    #     std_neg = returns[returns<0].std()*np.sqrt(N)
    #     return returns.mean()*N/std_neg

    # def calmar(returns,N,mdd):
    #     # > 0.5 is good, 3.0 to 5.0 is very good
    #     return returns.mean()*N/abs(mdd/100)
    
    # Fees
    fees = {}
    for c in summary_index:
        if c == "B":
            fees[c] = 0
        elif c in ["L","S"]:
            fees[c] = trades[f"{c}_fees"].sum()
        else:
            fees[c] = fees["L"]+fees["S"]
    
    # returns
    ret = {}
    for c in summary_index:
        final_ret = np.round(df[f"cum_{c}_pnl%"].iloc[-1]-100,2)
        ret[c]=final_ret
    
    # pnl and starting and final equity
    pnl = {} # cum_B_pnl
    eq_start = {}
    eq_end = {}
    for c in summary_index:
        if c == "B":
            pnl[c] = df["cum_B_pnl"].iloc[-1]
            eq_start[c] = total_equity
            eq_end[c] = eq_start[c] + pnl[c]
        elif c in ["L","S"]:
            final_pnl = df[f"cum_{c}_rpnl"].iloc[-1]
            if np.isnan(final_pnl):
                final_pnl = 0
            pnl[c]=final_pnl
            if c =="L":
                eq_start[c] = long_starting_equity
            else:
                eq_start[c] = short_starting_equity
            eq_end[c] = eq_start[c] + pnl[c]
        else:
            pnl[c]=pnl["L"]+pnl["S"]
            eq_start[c] = total_equity # eq_start["L"]+eq_start["S"]
            eq_end[c] = eq_start[c] + pnl[c]
    # maxdrawdown
    mdd = {}
    for c in summary_index:
        mdd[c] = np.round(df[f"dd_{c}"].min(),2)
    
    # time underwater
            
            
    summary = {"Sharpe": sharpes,
               "Total Return %":ret,
               "Equity Start $":eq_start,
               "Total Return $":pnl,
               "Fee (bps)": fee,
               "Total Fees $": fees,
               "Equity End $" :eq_end,
               # "":{"L":"","S":"","A":"","B":""},
               "Win Rate %":winrate,
               "Lose Rate %":loserate,
               "Draw Rate %": drawrate, 
               "pnl_mean $":pnl_mean,
               "pnl_median $":pnl_median,
               "wins_mean $":avg_wins,
               "wins_median $":med_wins,
               "loss_mean $":avg_loss,
               "loss_median $":med_loss,
               # "":{"L":"","S":"","A":"","B":""},
               "Profit Factor":p2g,
               "total_trades": total_trades,
               "MDD %":mdd,
               f"HP mean ({timeframe_interval})":time_in_trade_mean,
               f"HP max ({timeframe_interval})":time_in_trade_max,
               f"HP min ({timeframe_interval})":time_in_trade_min}
    
    summary_rounds = {"Sharpe": 2,
                       "Total Return %":2,
                       "Equity Start":2,
                       "Total Return":2,
                       "Fee (bps)": 1,
                       "Total Fees $": 2,
                       "Equity End " :2,
                       # "":{"L":"","S":"","A":"","B":""},
                       "Win Rate %":3,
                       "Lose Rate %":3,
                       "Draw Rate %": 3, 
                       "pnl_mean ":3,
                       "pnl_median":3,
                       "wins_mean":3,
                       "wins_median":3,
                       "loss_mean":3,
                       "loss_median":3,
                       # "":{"L":"","S":"","A":"","B":""},
                       "Profit Factor":2,
                       "total_trades": 1,
                       "MDD %":2,
                       "Dur median":0,
                       "Dur max":0,
                       "Dur min":0}
    
    # Tidy up trades table
    output_trade_cols = ['L_id',
                        'L_positions',
                        'L_entry_price',
                        'L_cost',
                        'L_qty',
                        'L_exit_price',
                        'L_rpnl',
                        'L_fees',
                        'S_id',
                        'S_positions',
                        'S_entry_price',
                        'S_cost',
                        'S_qty',
                        'S_exit_price',
                        'S_rpnl',
                        'S_fees',
                        'A_rpnl',
                        'cum_A_pnl']
    
    #  Tidy up summary tables
    summary_df = pd.DataFrame(summary)
    for col in summary.keys():
        summary_df[col] = summary_df[[col]].round(decimals=pd.Series(list(summary_rounds.values()), index=summary_df.T.index))
        # [2,2,2,2,2,2,2,3,3,3,3,3,3,1,1,2,2,2,2,2,2]
    summary_df = summary_df.astype(object).T
    
    summary_df=summary_df[["A","L","S","B"]]
    summary_df.rename(columns={"L":"longs only","S":"shorts only","A":"total","B":"buyhold"},inplace=True)
    return df, trades[output_trade_cols], summary_df



#%%
if __name__ == "__main__":
    x = 365*8
    y = 252 * 24
    print(y/x)
# %%
