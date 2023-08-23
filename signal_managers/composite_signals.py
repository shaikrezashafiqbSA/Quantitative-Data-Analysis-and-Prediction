import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def get_rolling_window_indexes(df0,
                               rolling_start_date = "2020-01-01",
                               rolling_window_period = "Y"):
    df = df0[rolling_start_date:].copy()
    df["rolling_window_period"] = df.index.to_period(rolling_window_period)
    df["date_time"] = df.index
    df = df.groupby(df["rolling_window_period"]).last()
    return list(df["date_time"])




def calc_rolling_composite_signal(sig_dict, 
                                  factors_to_compose = ["z_close_ES_USD", "spread_q"],
                                  factors_to_normalise = ["spread_q"], 
                                  test_factor_loadings = [0,1], # None
                                  rolling_start_date = "2020-01-01", 
                                  rolling_window_period = "Q",
                                  min_period = 1, 
                                  verbose = False,
                                  plot = False):
    
    if len(factors_to_compose) == 0:
        return sig_dict

    weights_dicts = {}
    for timeframe, df in sig_dict.items():
        weights_dict = {}
        normed_factor_list = []
        sig_pca_list = []
        scaler = None
        
        idxs = get_rolling_window_indexes(df,
                                          rolling_start_date = rolling_start_date, 
                                          rolling_window_period=rolling_window_period)
        if verbose: print(f"timeframe --------> {timeframe} \n {idxs} \n")
        # =========
        # START roll forward to calculate pca weights every quarter starting from 2021
        # =========
        for i in range(len(idxs)):
            end_idx = idxs[i]
            if min_period is None: # expanding window 
                df_fit = df[:end_idx].copy()
            else:
                if i >= min_period:
                    start_idx = idxs[i-min_period]
                    df_fit = df[start_idx:end_idx].copy()
                else:
                    df_fit = df[:end_idx].copy()


            ## NORMALISE
            # Normalise spreads 
            
            if factors_to_normalise is not None:
                for factor_to_normalise in factors_to_normalise:
                    if scaler is None:
                        # No scalar initialised so fit-transform first rolling window
                        scaler = StandardScaler()
                        df_fit[factor_to_normalise] = scaler.fit_transform(df_fit[[factor_to_normalise]])
                        if verbose: print(f"fit transforming window: {df_fit.index[0]} --> {df_fit.index[-1]}")
                    else:
                        # HAVE TO scaler.transform future factor_to_normalise instead of fit_transform every window
                        if verbose: print(f"transforming window: {df_fit.index[0]} --> {df_fit.index[-1]}")
                        df_fit[factor_to_normalise] = scaler.transform(df_fit[[factor_to_normalise]])
                        # scaler.fit(df_fit[[factor_to_normalise]])
                    normed_factor_list.append(df_fit[[factor_to_normalise]])
            
            
            ## CALCULATE factor loadings
            pca=PCA()
            np_sig_pca = pca.fit_transform(df_fit[factors_to_compose].dropna())
            # print(np.shape(np_sig_pca))
            #pca.components_ has the meaning of each principal component, 
            # essentially how it was derived 
            #checking shape tells us it has 2 rows, one for each principal component and 
            # 2 columns proportion of each of the 2 features for each row 
            
            loadings= pca.components_
            # Check loadings first
            if test_factor_loadings is None:
                for pc in loadings:
                    if np.sum(pc)>0:
                        factor_loadings = pc
                        break
            else:
                factor_loadings = test_factor_loadings
            # factor_loadings = loadings[0,:] #loadings[:,0]

            if verbose: print(f" =====> factor loadings {factors_to_compose}:\n {factor_loadings}\n=====>  full eigenvectors:\n {loadings}\n")
            temp = {}
            for factor_loading,factor_to_compose in zip(factor_loadings,factors_to_compose):
                temp[f"{factor_to_compose}_weight"] = factor_loading
            weights_dict[end_idx] = temp
        
        # =========
        # END ROLL FORWARD    
        # =========
        
        # collect all normed factor (assume 1 for this case)
        for factor_to_normalise in factors_to_normalise:
            normed_factor_df = pd.concat(normed_factor_list,axis=0)
            normed_factor_df.columns = [f"{factor_to_normalise}_norm"]
            normed_factor_df = normed_factor_df[~normed_factor_df.index.duplicated(keep='first')]
            if verbose: print(f"normed_df has duplicates? ---> {normed_factor_df.index.has_duplicates}")
            
            if verbose: print(f"df has duplicates? ---> {df.index.has_duplicates}")
            df = pd.merge(df, normed_factor_df, left_index=True, right_index=True, how="left")
            if verbose: print(f"df merged with normed_df has duplicates? ---> {df.index.has_duplicates}")
        # Edit factors_to_comp list to highlight which factors has been normalised
        for idx,item in enumerate(factors_to_compose):
            if item in factors_to_normalise:
                item = f"{item}_norm"
                factors_to_compose[idx] = item
            
        
        # Merge weights to original df
        weights_df = pd.DataFrame(weights_dict).T
            # merge to df 
        df = pd.merge(df, weights_df, left_index=True, right_index=True, how="left")
        if verbose: print(f"df merged with weights_df has duplicates? ---> {df.index.has_duplicates}")
        # foward fill weights to be used 
        factor_loadings_cols = list(weights_df.columns)
        df[factor_loadings_cols] = df[factor_loadings_cols].fillna(method="ffill")
        df[factor_loadings_cols] = df[factor_loadings_cols].fillna(method="bfill")
        
        # Multiply and sum weights with signal to form composite signal
        # comp_sig = pd.DataFrame()
        # comp_sig_names = []
        # for factor_to_compose in factors_to_compose:

        #     comp_sig[f"comp_{factor_to_compose}"] = df[factor_to_compose].multiply(df[f"{factor_to_compose}_weight"], axis='index',fill_value = None)
        #     comp_sig_names.append(f"comp_{factor_to_compose}")
        #     if timeframe == "5Min":
        #         comp_sig_output = comp_sig.copy()
        # df["comp_sig"] = comp_sig.sum(axis=1,min_count=1)
        
        df["comp_sig"] = (df.loc[:,factors_to_compose]*df.loc[:,factor_loadings_cols].values).sum(axis=1,min_count=len(factors_to_compose))
        
        # df["comp_sig"].fillna(0,inplace=True)
        # df.drop(columns=comp_sig_names,inplace=True)
        df.index.name = "date_time"
        sig_dict[timeframe] = df
        weights_dicts[timeframe] = weights_df
        
        
        # if plot:
        #     nrows = len(sig_dict)
        #     fig, axs = plt.subplots(nrows=nrows,ncols=1, sharex = 'col',figsize=(10,15))
        #     for timeframe,df in sig_dict.items():
        #         df[]
    return sig_dict
    


# Non rolling

def calc_composite_signal(sig_dict, 
                          factors_to_compose = ["spread_USDZ2022","z_close_ES1!"],
                          factors_to_normalise = ["spread_USDZ2022"], 
                          factor_loadings = {"5m":[0.99976268, -0.02178511]}, 
                          scaler = None,
                          verbose = False):

    for timeframe, df in sig_dict.items():
        if scaler is not None:
            print(df[factors_to_normalise].columns)
            df[factors_to_normalise] = scaler.transform(df[factors_to_normalise])
        df["comp_sig"] =  (df.loc[:,factors_to_compose]*factor_loadings[timeframe]).sum(axis=1, min_count=len(factors_to_compose))
        
        df.index.name = "date_time"
        sig_dict[timeframe] = df
        
    return sig_dict
    
    
        


def merge_signals(sig_dict, smallest_timeframe="5Min", target_signals=["comp_sig"]):
    df= sig_dict[smallest_timeframe]

    signals_to_query = [target_signals]+ ["L_uq", "L_lq", "S_uq", "S_lq"]
    
    df = df.add_prefix(f"{smallest_timeframe}_")
    for timeframe, df_i in sig_dict.items():
        if timeframe == smallest_timeframe:
            continue
        if "L_uq" not in df_i.columns:
            df = pd.merge(df, df_i[target_signals].add_prefix(f"{timeframe}_"), how="left", left_index=True, right_index=True)
        else:
            df = pd.merge(df, df_i[signals_to_query].add_prefix(f"{timeframe}_"), how="left", left_index=True, right_index=True)
        
    return df

def full_merge(sig_dict):
    pass