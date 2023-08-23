import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

def cross_acf_plots(instruments_dict,
                    Y={"instrument":"THB_USD",
                       "col":"close"
                       },
                    X={"instruments":['ES_USD',
                                      'TNX_USD',
                                      'EUR_USD'],
                       "col":"close"
                       },
                    apply_pct_change= {"X":True, "Y":False},
                    window=["2020-01-01","2022-12-31"],         
                    acf_max_lag = 40,   
                    granger_max_lag=1,
                    fig_width = 18,
                    timeframe = "h",
                    verbose=False):

    test_type = "ssr_ftest"

    #%%
    # assert Y and X in df_ret.columns
    # xlabels=np.arange(-acf_max_lag,acf_max_lag+1)
    # xlabels[xlabels%2==0]
    t = list(instruments_dict[Y['instrument']].keys())
    t = [int(i.split(f'{timeframe}')[0]) for i in t]
    Len_X = len(X["instruments"])
    if Len_X == 1:
        nrows = 2
    else:
        nrows = Len_X
    fig, axs = plt.subplots(nrows=nrows, 
                            ncols=len(t)+2, 
                            # sharex="col",
                            figsize=(fig_width,3*nrows),
                            gridspec_kw={'height_ratios': [1]*nrows, 
                                         'width_ratios': [1]*len(t) + [1,1]
                                         }
                            )
    
    lagged_correlation_df_dict = {}
    granger_df_dict = {}
    
    Y_instrument = Y["instrument"]
    Y_col = Y["col"]
    x_col = X["col"]
    
    for row_index,x in tqdm(enumerate(X["instruments"])):
        if verbose: print(f"======================== ROW {row_index} | {x} ======================== ")
        resampled_dict = {}
        lagged_correlation_df_list = []
        granger_dict = {}
        
        # ================================================================
        # Loop through timeframes
        # ================================================================
        for col_index,freq in enumerate(t): #col_index is used as 
            if verbose: print(f"============ COL {col_index} ============ ")
            
            df_Y = instruments_dict[Y_instrument][f"{freq}{timeframe}"][window[0]:window[1]]
            df_Y= df_Y[[Y_col]].copy()
            df_Y.rename(columns={Y_col:"Y"}, inplace=True)
            
            df_x = instruments_dict[x][f"{freq}{timeframe}"][window[0]:window[1]].copy()
            df_x = df_x[[x_col]].copy()
            df_x.rename(columns={x_col:"x"}, inplace=True)
            
            if apply_pct_change["Y"]:
                df_Y["Y"] = df_Y["Y"].pct_change()
            if apply_pct_change["X"]:
                df_x["x"] = df_x["x"].pct_change()
            
            df = pd.merge(df_Y,df_x, left_index=True,right_index=True, how="left")

            # ================================
            # Granger causality calculation
            # ================================
            try:
                test_df = df[["Y","x"]].copy()#.dropna()
                res=grangercausalitytests(test_df.dropna(), maxlag=granger_max_lag,verbose=False)
            except Exception as e:
                print(f"ERROR for {x}:\n{e} \nlen test_df: {len(test_df)}")
                return test_df
            
            
            p_values = {}
            for m in range(granger_max_lag):
                p_values[(f"{x}->{Y_instrument}",f"lag_{m+1}")] = np.round(res[m+1][0][test_type][1],3)
            
            # Y to x
            res1=grangercausalitytests(df[["x","Y"]].dropna(), maxlag=granger_max_lag,verbose=False)
            for m in range(granger_max_lag):
                p_values[(f"{Y_instrument}->{x}",f"lag_{m+1}")] = np.round(res1[m+1][0][test_type][1],3)
            granger_dict[freq]=p_values
    
                
            # ================================
            # XACF calculation
            # ================================
            if verbose: print(f"plotting XACF --> {freq} ({row_index},{col_index})")
            lagged_correlation = {f"{freq}{timeframe}": [df["x"].corr(df["Y"].shift(t)) for t in range(-acf_max_lag,acf_max_lag)]}
            lagged_correlation = pd.DataFrame.from_dict(lagged_correlation)
            lagged_correlation.index.name = "lags"
            lagged_correlation.index = lagged_correlation.index - acf_max_lag
            lagged_correlation_df_list.append(lagged_correlation)
            
            # ~~~~~~~~~~~~~~~~
            # START Plot ACF
            # ~~~~~~~~~~~~~~~~
            lagged_correlation.plot(ax=axs[row_index,col_index], kind='bar').axvline(acf_max_lag,color="red", alpha=0.5, linestyle = 'dashed')
    
            for index, label in enumerate(axs[row_index,col_index].xaxis.get_ticklabels()):
                if index%2 != 0:
                    label.set_visible(False)
                    
            axs[row_index,0].set_title(f"XACF of {x} {X['col']} with lagged {Y['instrument']} {Y['col']}",fontsize=9)
            # ~~~~~~~~~~~~~~~~
            # END Plot ACF
            # ~~~~~~~~~~~~~~~~
            
            
            
        # ~~~~~~~~~~~~~~~~
        # START PLOT GRANGER
        # ~~~~~~~~~~~~~~~~
        gplot1_col_index = -2
        gplot2_col_index = -1
        axs[row_index,gplot1_col_index].axis("off")
        axs[row_index,gplot2_col_index].axis("off")
        
    
        granger_plot_df = pd.DataFrame.from_records(granger_dict).T
        xy_df = granger_plot_df[[f"{x}->{Y_instrument}"]]
        yx_df = granger_plot_df[[f"{Y_instrument}->{x}"]]
            
    
        # PLOT TABLE for x ==> Y
        if verbose: print(f"============ COL {col_index+1} ============ ")
        if verbose: print(f"granger: {x} -> {Y}")
        # Conditional table coloring for high sig
        colors_xy = []
        for _, row in xy_df.iterrows():
            colors_in_column = ["r"]*len(xy_df.columns)
            for i,col in enumerate(xy_df.columns):
                if row[col]<=0.05:
                    colors_in_column[i] = "g"
            colors_xy.append(colors_in_column)
            
        # granger table  
        axs[row_index,gplot1_col_index].table(cellText=xy_df.values,
                       rowLabels=[f"{t}{timeframe} ret" for t in xy_df.index],
                       colLabels=[f"lag {i}" for i in range(1,granger_max_lag+1)],
                       loc="center",
                       cellColours=colors_xy
                       )
        # axs[row_index,gplot1_col_index].title.set_text(f"{x} -> {Y} pval")
        axs[row_index,gplot1_col_index].set_title(f"\n\n\n{x} ({X['col']})->{Y_instrument} ({Y['col']})\n p values",fontsize=9)
        
        
        
        # PLOT TABLE for Y ==> x
        if verbose: print(f"============ COL {col_index+2} ============ ")
        if verbose: print(f"granger: {Y} -> {x}")
        # Conditional table coloring for high sig
        colors_yx = []
        for _, row in yx_df.iterrows():
            colors_in_column = ["r"]*len(yx_df.columns)
            for i,col in enumerate(yx_df.columns):
                if row[col]<=0.05:
                    colors_in_column[i] = "g"
            colors_yx.append(colors_in_column)
            
        # granger table    
        axs[row_index,gplot2_col_index].table(cellText=yx_df.values,
                       rowLabels=[f"{t}{timeframe} ret" for t in yx_df.index],
                       colLabels= [f"lag {i}" for i in range(1,granger_max_lag+1)],
                       loc="center",                           
                       cellColours=colors_yx
                      )
        # axs[row_index,gplot2_col_index].title.set_text(f"{Y} -> {x} pval")
        axs[row_index,gplot2_col_index].set_title(f"\n\n\n{Y_instrument} ({Y['col']}) ->{x} ({X['col']})\n p values",fontsize=9)
            
        # ~~~~~~~~~~~~~~~~
        # END Plot GRANGER
        # ~~~~~~~~~~~~~~~~
            
        
        # ================================
        # Collect xcross data  
        # ================================
        lagged_correlation_df = pd.concat(lagged_correlation_df_list,axis=1)
        granger_df = pd.DataFrame.from_records(granger_dict).T
        granger_df.set_index(granger_df.index.astype(str) + ' mins',inplace=True)
        
        lagged_correlation_df_dict[x] = lagged_correlation_df
        granger_df_dict[x] = granger_df
        
        # resize ylims according to maxmin total
        max_y = np.round(lagged_correlation_df.max().max(),2)
        min_y = np.round(lagged_correlation_df.min().min(),2)
        for i,freq in enumerate(t):
            axs[row_index,i].set_ylim(min_y,max_y)
        
        # set labels
        plt.setp(axs[row_index, :-2], xlabel='LAG')
        plt.setp(axs[row_index, :-2], ylabel='Corr')
    
    
    
    
    ### END
    fig.suptitle(f"lead-lag analysis of {X['instruments']} ==> {Y['instrument']}\n({X['col']} ==> {Y['col']}) \n{df_Y.index[0].strftime('%Y-%m-%d')} to {df_Y.index[-1].strftime('%Y-%m-%d')}")
    plt.tight_layout()
    
    return lagged_correlation_df_dict,granger_df_dict
    
    
    