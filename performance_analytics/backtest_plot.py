import matplotlib.pyplot as plt
from signal_managers import rolling_sr
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

def backtest_plots(df,
                   metrics_table0,
                   horizon_labels=None,
                   show_B=True,
                   title="test", 
                   figsize=(20,12), 
                   linewidth=0.75, 
                   fees=0.0002):
    
    
    fig, axs = plt.subplots(nrows=6,ncols=2, sharex = False,gridspec_kw={'height_ratios': [4,1,2,1,1,1], 'width_ratios':[3,2]},figsize=figsize)
    
    alpha=0.5
    
    # ==========================================================================================================================================================
    # Column 0
    # ==========================================================================================================================================================
    
    # =============================================================================
    # % pnl plots
    # =============================================================================
    subplot_row = 0
    subplot_col = 0
    long_only_label = "long only"
    short_only_label = "short only"
    strat_label = "total"
    if show_B:
        df["cum_B_pnl%"].plot(label=f'buyhold % benchmark', color="black",ax=axs[subplot_row,subplot_col], linewidth=linewidth)
    if horizon_labels is None:
        df['cum_A_pnl%'].plot(label=strat_label, color="blue",ax=axs[subplot_row,subplot_col], linewidth=linewidth)
        df['cum_L_pnl%'].plot(label=long_only_label, color="green",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)
    elif len(horizon_labels)==1:
        df['cum_A_pnl%'].plot(label=strat_label, color="blue",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axvline(x=horizon_labels[0], color='red', ls="--")
        df['cum_L_pnl%'].plot(label=long_only_label, color="green",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)
    else:
        df['cum_A_pnl%'].plot(label=strat_label, color="blue",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axvline(x=horizon_labels[0], color='red', ls="--")
        df['cum_L_pnl%'].plot(label=long_only_label, color="green",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axvline(x=horizon_labels[1], alpha=alpha, color='green',ls="--")
        
    df['cum_S_pnl%'].plot(label=short_only_label, color="orange",ax=axs[subplot_row,subplot_col], alpha=alpha, linewidth=linewidth).axhline(100,color="red")
    axs[subplot_row,subplot_col].legend(loc='center left')    
    axs[subplot_row,subplot_col].title.set_text(f"% PNL Returns")
    
    
    # =============================================================================
    # Realised scatters
    # =============================================================================
    
    # -----------------------
    # full strat POSITION SIZE scatter
    # -----------------------
    subplot_row = 5
    subplot_col = 0
    
    df.reset_index().dropna(subset=["A_rpnl"]).plot.scatter(x='date_time',
                                                            y='A_rpnl',
                                                            color="blue",
                                                            ax=axs[subplot_row,subplot_col],
                                                            s=0.5).axhline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Realised Pnl Scatter")
    
    # -----------------------
    # long side POSITION SIZE scatter
    # -----------------------
    subplot_row = 3
    subplot_col = 0
    
    df.reset_index().dropna(subset=["L_rpnl"]).plot.scatter(x='date_time',
                                                            y='L_cost',
                                                            color="green",
                                                            ax=axs[subplot_row,subplot_col],
                                                            s=0.5).axhline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Long position sizes")
    
    # -----------------------
    # short side POSITION SIZE scatter
    # -----------------------
    subplot_row = 4
    subplot_col = 0
    
    df.reset_index().dropna(subset=["S_rpnl"]).plot.scatter(x='date_time',
                                                            y ='S_cost',
                                                            color="orange",
                                                            ax=axs[subplot_row,subplot_col],
                                                            s=0.5).axhline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Short position sizes")

    
    
    
    # =============================================================================
    # Drawdowns
    # =============================================================================
    
    subplot_row = 1
    subplot_col = 0
    
    
    long_only_label = "long only daily drawdown"
    short_only_label = "short only daily drawdown"
    strat_label = "total % daily drawdown"
    
    # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    # groupby(df.index.date).max().plot(label=f'daily max drawdown %', color="red",ax=axs[1,1], linewidth=linewidth)
    try:
        df['dd_A'].groupby(df.index.date).max().plot(label=strat_label, color="blue",ax=axs[subplot_row,subplot_col], linewidth=linewidth)
        df['dd_L'].groupby(df.index.date).max().plot(label=long_only_label, color="green",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)
        df['dd_S'].groupby(df.index.date).max().plot(label=short_only_label, color="orange",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)#.axhline(100,color="red")
        axs[subplot_row,subplot_col].title.set_text(f"% Daily Max Drawdowns")
    except Exception as e:
        print(f"Drawdown plots bugged - {e}")
        
    
    
    # =============================================================================
    # Rolling Sharpe
    # =============================================================================
    subplot_row = 2
    subplot_col = 0
    
    # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    df['A_rolling_SR'].plot(label=f'total', color="blue",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axhline(0,color="red")
    
    df['L_rolling_SR'].plot(label=f'long only', color="green",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)
    df['S_rolling_SR'].plot(label=f'short only', color="orange",ax=axs[subplot_row,subplot_col], linewidth=linewidth, alpha=alpha)
    # axs[5,0].legend(loc='center left', bbox_to_anchor=(4, 0.5))
    # axs[5,0].set_ylim([-5, 0])
    axs[subplot_row,subplot_col].title.set_text(f"Quarterly rolling sharpes")
    
    
    
    # ==========================================================================================================================================================
    # Column 1
    # ==========================================================================================================================================================
    #  metrics tables
    

            
    subplot_row = 0
    subplot_col = 1
    metrics_table = metrics_table0.copy()
    # if fees == 0.0:
    if fees ==0.0:
        metrics_table=metrics_table.drop(["Fee (bps)","Total Fees $"])
    colors_yx = []
    for metric, row in metrics_table.iterrows():
        colors_in_column = ["#DCDCDC"]*len(metrics_table.columns)
        for i,col in enumerate(metrics_table.columns):
            if metric in ["Equity Start $", "Fee (bps)","total_trades", "Total Fees $", "Holding period median","Holding period max","Holding period Min"]:
                colors_in_column[i] = "#DCDCDC"
                
            elif metric == "Equity End $":
                if row[col] <=metrics_table.loc["Equity Start $",col]:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98"
                    
            elif metric == "Win Rate %":
                if row[col] <=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Lose Rate %":
                if row[col] >=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Profit Factor":
                if row[col] <=1:
                    colors_in_column[i] = "#CD5C5C" # red
                else:
                    colors_in_column[i] = "#98FB98"  # green
                    
            elif row[col]>0.0:
                colors_in_column[i] = "#98FB98"
            elif row[col] < 0.0:
                colors_in_column[i] = "#CD5C5C"
                
        colors_yx.append(colors_in_column)
        
    table = axs[subplot_row,subplot_col].table(cellText=metrics_table.values,
                           rowLabels=list(metrics_table.index),
                           colLabels= list(metrics_table.columns),
                           loc="right",#colWidths=[0.2,0.2,0.2]
                           cellColours=colors_yx,
                           bbox = [0.25, -0.5, 1, 1.5] # x, y shift, width, height
                           )
    
    
    # =============================================================================
    # realised histos
    # =============================================================================
    subplot_row = 5
    subplot_col = 1
    df['A_rpnl'].hist(bins=50,ax=axs[subplot_row,subplot_col],color="blue").axvline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Realised histo")
    
    subplot_row = 3
    subplot_col = 1
    df['L_rpnl'].hist(bins=50,ax=axs[subplot_row,subplot_col],color="green").axvline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Realised (L only) histo")
    
    subplot_row = 4
    subplot_col = 1
    df['S_rpnl'].hist(bins=50,ax=axs[subplot_row,subplot_col],color="orange").axvline(0,color="red")
    axs[subplot_row,subplot_col].title.set_text(f"Realised (S only) histo")
    
    axs_to_remove = [[0,1],[1,1],[2,1]]
    for ax_to_remove in axs_to_remove:
        axs[ax_to_remove[0],ax_to_remove[1]].axis("off")
    
    # set_share_axes(axs[:,0], sharex=True)
    fig.autofmt_xdate()
    fig.suptitle(f"{title}\n {df.index[0].date()} to {df.index[-1].date()}")
    # fig.tight_layout() 
    
    
    
 
def backtest_plots_simplified(df,
                              metrics_table0,
                               horizon_labels=None,
                               show_B=True,
                               title="test", 
                               figsize=(20,12), 
                               linewidth=0.75,
                               fees=0.0002,
                               kline_to_trade = "5m_close"):
    fig, axs = plt.subplots(nrows=5,ncols=2,sharex = False, gridspec_kw={'height_ratios': [3,3,3,3,3]},figsize=figsize) #'width_ratios':[3,2]
    alpha = 0.5 
    fig.set_dpi(300.0)
    # axs[].axis("off")
    # axs[2,1].axis("off")
    
    # =============================================================================
    # SUBPLOT 0: metrics table
    # =============================================================================
    subplot_row = 0
    subplot_col = 1
    axs[0,subplot_col].axis("off")
    # axs[0,subplot_col+1].axis("off")
    # axs[1,subplot_col+1].axis("off")
    metrics_table = metrics_table0.copy()
    if fees ==0.0:
        metrics_table=metrics_table.drop(["Fee (bps)","Total Fees $"])
    # metrics_table =metrics_table.T
    # plt.rcParams["figure.dpi"] = 300
    colors_yx = []
    for metric, row in metrics_table.iterrows():
        colors_in_column = ["#DCDCDC"]*len(metrics_table.columns)
        for i,col in enumerate(metrics_table.columns):
            if metric in ["Equity Start $", "Fee (bps)","total_trades", "Total Fees $", "Holding period median","Holding period max","Holding period Min"]:
                colors_in_column[i] = "#DCDCDC"
                
            elif metric == "Equity End $":
                if row[col] <=metrics_table.loc["Equity Start $",col]:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98"
                    
            elif metric == "Win Rate %":
                if row[col] <=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Lose Rate %":
                if row[col] >=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Profit Factor":
                if row[col] <=1:
                    colors_in_column[i] = "#CD5C5C" # red
                else:
                    colors_in_column[i] = "#98FB98"  # green
                    
            elif row[col]>0.0:
                colors_in_column[i] = "#98FB98"
            elif row[col] < 0.0:
                colors_in_column[i] = "#CD5C5C"
                
        colors_yx.append(colors_in_column)
        
    table = axs[subplot_row,subplot_col].table(cellText=metrics_table.values,
                           rowLabels=list(metrics_table.index),
                           colLabels= list(metrics_table.columns),
                           loc="right",#colWidths=[0.2,0.2,0.2]
                           cellColours=colors_yx,
                           bbox = [0.4, 0, 0.5, 2.2] # x, y shift, width, height
                           )
    # =============================================================================
    # SUBPLOT 0,0: % pnl
    # =============================================================================
    subplot_row = 1
    subplot_col = 0
    
    long_only_label = "long only"
    short_only_label = "short only"
    strat_label = "total"
    df[["cum_B_pnl%","cum_A_pnl%","cum_L_pnl%","cum_S_pnl%"]]=df[["cum_B_pnl%","cum_A_pnl%","cum_L_pnl%","cum_S_pnl%"]]-100


        
    
    # line plots
    if horizon_labels is None:
        df["cum_B_pnl%"].plot(label=f'buyhold % pnl', color="black",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    else:
        df["cum_B_pnl%"].plot(label=f'buyhold % pnl', color="black",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5).axvline(x=horizon_labels[0], color='red', ls="--")
    df['cum_A_pnl%'].plot(label=strat_label, color="blue",ax=axs[subplot_row+1,subplot_col], linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    df['cum_L_pnl%'].plot(label=long_only_label, color="green",ax=axs[subplot_row+2,subplot_col], linewidth=linewidth, alpha=alpha).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    df['cum_S_pnl%'].plot(label=short_only_label, color="orange",ax=axs[subplot_row+3,subplot_col], alpha=alpha, linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    
    # BAR PLOTS
    df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_B_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1]).plot(kind="bar",ax=axs[subplot_row,subplot_col+1],color="black").axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.5)#.axhline(0,color="red"
    df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_A_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1]).plot(kind="bar",ax=axs[subplot_row+1,subplot_col+1], color="blue").axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.5)#.axhline(0,color="red")
    df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_L_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1]).plot(kind="bar",ax=axs[subplot_row+2,subplot_col+1], color="green").axhline(color="red", linewidth=0.5, linestyle="--", alpha=0.5)#.axhline(0,color="red")
    df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_S_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1]).plot(kind="bar",ax=axs[subplot_row+3,subplot_col+1], color="orange").axhline(color="red", linewidth=0.5, linestyle="--", alpha=0.5)#.axhline(0,color="red")
    
    # axs[subplot_row].legend(loc='center left')    
    # for i in range(1,10):
    #     axs[i].grid(True, linestyle='--')
        
    axs[subplot_row,subplot_col].title.set_text(f"% Buyhold benchmark BTCUSD returns")
    axs[subplot_row+1,subplot_col].title.set_text(f"% MODEL BTCUSD returns")
    # axs[subplot_row+2,subplot_col].set(xticklabels=[])
    
    axs[subplot_row+2,subplot_col].title.set_text(f"% MODEL BTCUSD (LONG ONLY) returns")
    # axs[subplot_row+4,subplot_col].set(xticklabels=[])
    
    axs[subplot_row+3,subplot_col].title.set_text(f"% MODEL BTCUSD (SHORT ONLY) returns")
    # axs[subplot_row+6,subplot_col].set(xticklabels=[])
    # axs[subplot_row].legend(bbox_to_anchor=(1.125,1))
    
    # for row in [subplot_row,subplot_row+1,subplot_row+2,subplot_row+3][:1]:
    #     print(f"date formatting for row,col: {row},{subplot_col}")
    #     date_fmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    #     date_loc = mdates.AutoDateLocator()
    #     axs[row,subplot_col].xaxis.set_major_formatter(date_fmt)
    #     axs[row,subplot_col].xaxis.set_major_locator(date_loc)
        
    #     # axs[row,subplot_col].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    #     # axs[row,subplot_col].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    #     # # set font and rotation for date tick labels
    #     # # plt.gcf().autofmt_xdate()
    #     for label in axs[row,subplot_col].get_xticklabels(which='major'):
    #         label.set(rotation=90, horizontalalignment='right')
    
    # =============================================================================
    # SUBPLOT 4,0 drawdown
    # =============================================================================
    # subplot_row = 2
    # subplot_col = 0
    
    # # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    # df['dd_A'].groupby(df.index.date).max().plot(label=f'daily max drawdown %', color="red",ax=axs[subplot_row,subplot_col], linewidth=linewidth)
    # # df[['dd_L']].plot(label=f'L%', color="green",ax=axs[2], linewidth=linewidth)
    # # df[['dd_S']].plot(label=f'S%', color="orange",ax=axs[2], linewidth=linewidth).axhline(100,color="red")
    # # axs[4,0].legend(loc='center left', bbox_to_anchor=(4, 0.5))
    # # axs[4,0].set_ylim([-5, 0])
    # axs[subplot_row,subplot_col].title.set_text(f"% Daily Max Drawdowns")
    
    
    
    # =============================================================================
    # market plots
    # =============================================================================
    subplot_row = 0
    subplot_col = 0
    
    timeframe= kline_to_trade.split("_")[0]
    test= df[[f"{timeframe}_close_US500", f"{timeframe}_close"]].copy()
    test.rename(columns={f"{timeframe}_close_US500":"SPX",f"{timeframe}_close":"BTCUSD"},inplace=True)
    test.plot(secondary_y="SPX",ax=axs[subplot_row,subplot_col],linewidth=linewidth)#,bbox = [0.4, -0.5, 0.4, 4]) # x, y shift, width, height)
    axs[subplot_row,subplot_col].title.set_text(f"SPX and BTCUSD prices")
    
    # axs[subplot_col, subplot_col].
    # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    # df['rolling_SR'].plot(label=f'rolling quarterly sharpe', color="green",ax=axs[2,1], linewidth=linewidth).axhline(0,color="red")
    # # axs[5,0].legend(loc='center left', bbox_to_anchor=(4, 0.5))
    # # axs[5,0].set_ylim([-5, 0])
    # axs[2,1].title.set_text(f"Quarterly rolling sharpe")
    # axs[subplot_row].legend(bbox_to_anchor=(1.3,0.6))
    # plt.savefig(f"./backtests/{timeframe}BTCSPX_model.png", dpi=300, bbox_inches="tight")
    fig.suptitle(f"{title}\n {df.index[0].date()} to {df.index[-1].date()}")
    fig.tight_layout(pad=0., w_pad=0.0, h_pad=0.1)
    
    
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
def backtest_plots_ppt(df,
                       df_trades,
                        metrics_table0,
                         horizon_labels=None,
                         show_B=True,
                         show_LS=False,
                         title="test", 
                         figsize=(25,12), 
                         linewidth=0.75,
                         fees=0.0002,
                         equity=1e6,
                         kline_to_trade = "5m_close",
                         file_name = "model",
                         to_drop=["Lose Rate %"]):
    

    fig = plt.figure()
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    fig.set_dpi(300.0)
    
    ax_key_pnl = plt.subplot2grid(shape=(4,6), loc=(0,0), colspan=6)
    ax_table = plt.subplot2grid(shape=(4,6), loc=(1,5), rowspan=2, colspan=2)
    
    ax_key_bins = plt.subplot2grid(shape=(4,6), loc=(1,0), colspan=4)
    ax_key_bins2 = plt.subplot2grid(shape=(4,6), loc=(2,0), colspan=4)
    ax_key_bins3 = plt.subplot2grid(shape=(4,6), loc=(3,0), colspan=4)
    
    ax_pct_aum = plt.subplot2grid(shape=(4,6), loc=(3,4), colspan=4)

        # print(f"========????? bins_df['%']{bins_df}")

    # ax_key_pnl = plt.subplot2grid(shape=(4,6), loc=(1,0), colspan=6)
    # ax_table = plt.subplot2grid(shape=(4,6), loc=(0,0), colspan=6)
    
    # ax_key_bins = plt.subplot2grid(shape=(4,6), loc=(1,0), colspan=6)
    # ax_key_bins2 = plt.subplot2grid(shape=(4,6), loc=(2,0), colspan=6)
    # ax_key_bins3 = plt.subplot2grid(shape=(4,6), loc=(3,0), colspan=6)
    
    # ax_pct_aum = plt.subplot2grid(shape=(4,6), loc=(3,4), colspan=4)

    
    
    alpha = 0.5 

    
    # =============================================================================
    # SUBPLOT 0: metrics table
    # =============================================================================
    subplot_row = 0
    subplot_col = 1
    ax_table.axis("off")


    metrics_table = metrics_table0.copy()
    metrics_table.rename(columns={"buyhold":"benchmark"},inplace=True)
    if fees ==0.0:
        metrics_table=metrics_table.drop(["Fee (bps)","Total Fees $"])
    for row_to_drop in to_drop:
        metrics_table.drop(row_to_drop,inplace=True)
    # plt.rcParams["figure.dpi"] = 300
    hp_labels = list(metrics_table.T.filter(regex="HP").columns)
    colors_yx = []
    for metric, row in metrics_table.iterrows():
        colors_in_column = ["#DCDCDC"]*len(metrics_table.columns)
        for i,col in enumerate(metrics_table.columns):
            if metric in ["Equity Start $", "Fee (bps)","total_trades", "Total Fees $"]+hp_labels:
                colors_in_column[i] = "#DCDCDC"
                
            elif metric == "Equity End $":
                if row[col] <=metrics_table.loc["Equity Start $",col]:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98"
                    
            elif metric == "Win Rate %":
                if row[col] <=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Lose Rate %":
                if row[col] >=50:
                    colors_in_column[i] = "#CD5C5C"
                else:
                    colors_in_column[i] = "#98FB98" 
                    
            elif metric == "Profit Factor":
                if row[col] <=1:
                    colors_in_column[i] = "#CD5C5C" # red
                else:
                    colors_in_column[i] = "#98FB98"  # green
                    
            elif row[col]>0.0:
                colors_in_column[i] = "#98FB98"
            elif row[col] < 0.0:
                colors_in_column[i] = "#CD5C5C"
                
        colors_yx.append(colors_in_column)
        
    table = ax_table.table(cellText=metrics_table.values,
                           rowLabels=list(metrics_table.index),
                           colLabels= list(metrics_table.columns),
                           loc="right",#colWidths=[0.2,0.2,0.2]
                           cellColours=colors_yx,
                           bbox = [-0.5, 0, 1.7, 1] # x, y shift, width, height
                           )
    table.set_fontsize(20)
    table.scale(2, 2)  # may help
    
    ax_table.set_title(f"(C) Key Performance Metrics",y=1,x=0,fontdict={"fontsize":20,"color":"darkblue"})
    
    
    
    
    
    
    # =============================================================================
    # ax_key_pnl 
    # =============================================================================

    
    long_only_label = "long only"
    short_only_label = "short only"
    strat_label = "total"
    df[["cum_B_pnl%","cum_A_pnl%","cum_L_pnl%","cum_S_pnl%"]]=df[["cum_B_pnl%","cum_A_pnl%","cum_L_pnl%","cum_S_pnl%"]]-100


        
    

    """
    a) Key performance: model vs index
    """
    test= df[[f"cum_B_pnl%", f"cum_A_pnl%",f"cum_L_pnl%",f"cum_S_pnl%"]].copy()
    test.rename(columns={f"cum_B_pnl%":"Benchmark",f"cum_A_pnl%":"Model",f"cum_L_pnl%":"longs",f"cum_S_pnl%":"shorts" },inplace=True)
    test.index.name=None
    if show_LS and show_B:
        lines_to_plot = ["Benchmark","longs","shorts"]
        color = ["black","green","orange"]
        style = ["--","-","-"]
    
    elif not show_LS and show_B:
        lines_to_plot = ["Benchmark"]
        color=["black","blue"]
        style = ["--","-","-"]

    elif not show_LS and not show_B:
        lines_to_plot = []
        color=["black","blue"]
        style = ["--","-","-"]

    if len(lines_to_plot) > 0:
        if horizon_labels is None:    
            test[lines_to_plot].plot(ax=ax_key_pnl,linewidth=linewidth, color=color, style=style)
        else:
            test[lines_to_plot].plot(ax=ax_key_pnl,linewidth=linewidth, color=color, style=style).axvline(horizon_labels,color="red", linewidth=0.5, linestyle="-", alpha=1)

    
    test[["Model"]].plot(ax=ax_key_pnl, linewidth=linewidth,color=["blue"], style=["-"]).axhline(0,color="red", linewidth=0.5, linestyle="-", alpha=1)
        
    ax_key_pnl.set_title(f"{title}\n{df.index[0].date()} to {df.index[-1].date()}\n\n\n(A) Performance: % Excess Returns VS Benchmark",fontdict={"fontsize":20,"color":"darkblue"}, loc='left')
    ax_key_pnl.grid()
    # df["cum_B_pnl%"].plot(label=f'buyhold % pnl', color="black",ax=axs[subplot_row,subplot_col], linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    # df['cum_A_pnl%'].plot(label=strat_label, color="blue",ax=axs[subplot_row+1,subplot_col], linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    # df['cum_L_pnl%'].plot(label=long_only_label, color="green",ax=axs[subplot_row+2,subplot_col], linewidth=linewidth, alpha=alpha).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    # df['cum_S_pnl%'].plot(label=short_only_label, color="orange",ax=axs[subplot_row+3,subplot_col], alpha=alpha, linewidth=linewidth).axhline(0,color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    
    # =============================================================================
    # BAR PLOTS MONTHLY
    # =============================================================================
    bins_df = pd.DataFrame()
    bins_df["Benchmark"]=df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_B_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["Model"] = df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_A_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["longs"] = df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_L_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["shorts"] = df.groupby(pd.PeriodIndex(df.index, freq="M"))["cum_S_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df.index.name = None
    if show_LS:
        bins_df[["Benchmark","Model", "longs","shorts"]].plot(kind="bar",ax=ax_key_bins,color=["black","blue", "green","orange"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
    else:
        bins_df[["Benchmark","Model"]].plot(kind="bar",ax=ax_key_bins,color=["black","blue"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
        
    
    ax_key_bins.set_title(f"(B)(i) Performance: % Excess MONTHLY Returns VS Benchmark",fontdict={"fontsize":15,"color":"darkblue"},x=0.38)
    ax_key_bins.grid()
    ax_key_bins.tick_params(labelrotation=30, labelsize=8)
    # ax_key_bins.yaxis.set_major_locator(ticker.MultipleLocator(10)) 

        
    # =============================================================================
    # BAR PLOTS QTRLY
    # =============================================================================
    bins_df = pd.DataFrame()
    bins_df["Benchmark"]=df.groupby(pd.PeriodIndex(df.index, freq="Q"))["cum_B_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["Model"] = df.groupby(pd.PeriodIndex(df.index, freq="Q"))["cum_A_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["longs"] = df.groupby(pd.PeriodIndex(df.index, freq="Q"))["cum_L_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["shorts"] = df.groupby(pd.PeriodIndex(df.index, freq="Q"))["cum_S_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df.index.name = None
    if show_LS:
        bins_df[["Benchmark","Model", "longs","shorts"]].plot(kind="bar",ax=ax_key_bins2,color=["black","blue", "green","orange"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
    else:
        bins_df[["Benchmark","Model"]].plot(kind="bar",ax=ax_key_bins2,color=["black","blue"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
        
        
    ax_key_bins2.set_title(f"(B) (ii) Performance: % Excess QUARTERLY Returns VS Benchmark",fontdict={"fontsize":15,"color":"darkblue"},x=0.38)
    ax_key_bins2.grid()
    ax_key_bins2.tick_params(labelrotation=30, labelsize=8)
    # ax_key_bins2.yaxis.set_major_locator(ticker.MultipleLocator(10)) 


    # =============================================================================
    # BAR PLOTS Yearly
    # =============================================================================
    bins_df = pd.DataFrame()
    bins_df["Benchmark"]=df.groupby(pd.PeriodIndex(df.index, freq="Y"))["cum_B_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["Model"] = df.groupby(pd.PeriodIndex(df.index, freq="Y"))["cum_A_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["longs"] = df.groupby(pd.PeriodIndex(df.index, freq="Y"))["cum_L_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df["shorts"] = df.groupby(pd.PeriodIndex(df.index, freq="Y"))["cum_S_pnl%"].apply(lambda x: x.iloc[-1]-x.fillna(method="bfill").iloc[1])
    bins_df.index.name = None
    if show_LS:
        bins_df[["Benchmark","Model", "longs","shorts"]].plot(kind="bar",ax=ax_key_bins3,color=["black","blue", "green","orange"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
    else:
        bins_df[["Benchmark","Model"]].plot(kind="bar",ax=ax_key_bins3,color=["black","blue"]).axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
    ax_key_bins3.set_title(f"(B) (iii) Performance: % Excess YEARLY Returns VS Benchmark",fontdict={"fontsize":15,"color":"darkblue"},x=0.38)
    ax_key_bins3.grid()
    ax_key_bins3.tick_params(labelrotation=30, labelsize=8)
    # ax_key_bins3.yaxis.set_major_locator(ticker.MultipleLocator(10)) 
    
    # =============================================================================
    # SUBPLOT aum pct
    # =============================================================================
    # subplot_row = 2
    # subplot_col = 0
    
    # # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    # df['dd_A'].groupby(df.index.date).max().plot(label=f'daily max drawdown %', color="red",ax=axs[subplot_row,subplot_col], linewidth=linewidth)
    # # df[['dd_L']].plot(label=f'L%', color="green",ax=axs[2], linewidth=linewidth)
    # # df[['dd_S']].plot(label=f'S%', color="orange",ax=axs[2], linewidth=linewidth).axhline(100,color="red")
    # # axs[4,0].legend(loc='center left', bbox_to_anchor=(4, 0.5))
    # # axs[4,0].set_ylim([-5, 0])
    # axs[subplot_row,subplot_col].title.set_text(f"% Daily Max Drawdowns")
    
    
    
    # =============================================================================
    # AUM% plots
    # =============================================================================
    dft = df_trades.copy()
    dft["L_vol"] = np.where(dft["L_positions"]==1,dft["L_cost"], 0)
    dft["S_vol"] = np.where(dft["S_positions"]==-1,dft["S_cost"], 0)
    
    dft["vol_traded"] = dft[["L_vol","S_vol"]].sum(axis=1)
    dft["AUM"] = dft["cum_A_pnl"]+equity
    
    dft["vol_traded"] = dft[["L_vol","S_vol"]].sum(axis=1)
    
    bins_df = pd.DataFrame()
    
    bins_df["Total Traded Vol"] = dft.groupby(pd.PeriodIndex(dft.index, freq="Y"))["vol_traded"].sum()
    bins_df["AUM end"] = dft.groupby(pd.PeriodIndex(dft.index, freq="Y"))["AUM"].apply(lambda x: x.iloc[-1])
    bins_df["%"] = bins_df["Total Traded Vol"]/bins_df["AUM end"]*100 

    bins_df[["%"]].plot(kind="bar",ax=ax_pct_aum,color=["purple"])#.axhline(color="red",linewidth=0.5, linestyle="--", alpha=0.75)#.axhline(0,color="red"
    ax_pct_aum.tick_params(axis='both', which='major', labelsize=6)
    ax_pct_aum.tick_params(axis='both', which='minor', labelsize=4)
    ax_pct_aum.set_title(f"(D) Total Traded Volume as % of AUM",fontdict={"fontsize":15,"color":"darkblue"},x=0.45)
    # test= df[[f"{timeframe}_close_US500", f"{timeframe}_close"]].copy()
    # test.rename(columns={f"{timeframe}_close_US500":"SPX",f"{timeframe}_close":"BTCUSD"},inplace=True)
    # test.plot(secondary_y="SPX",ax=axs[subplot_row,subplot_col],linewidth=linewidth)#,bbox = [0.4, -0.5, 0.4, 4]) # x, y shift, width, height)
    # axs[subplot_row,subplot_col].title.set_text(f"SPX and BTCUSD prices")
    
    # axs[subplot_col, subplot_col].
    # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    # df['rolling_SR'].plot(label=f'rolling quarterly sharpe', color="green",ax=axs[2,1], linewidth=linewidth).axhline(0,color="red")
    # # axs[5,0].legend(loc='center left', bbox_to_anchor=(4, 0.5))
    # # axs[5,0].set_ylim([-5, 0])
    # axs[2,1].title.set_text(f"Quarterly rolling sharpe")
    # axs[subplot_row].legend(bbox_to_anchor=(1.3,0.6))
    # plt.savefig(f"./backtests/{timeframe}BTCSPX_model.png", dpi=300, bbox_inches="tight")
    # fig.suptitle(f"{title}\n {df.index[0].date()} to {df.index[-1].date()}", fontsize="40")
    # fig.tight_layout(pad=0., w_pad=0.0, h_pad=0.1)
    # if file_name is not None:
    #     fig.savefig(f"./backtests/{file_name}.pdf", bbox_inches='tight')