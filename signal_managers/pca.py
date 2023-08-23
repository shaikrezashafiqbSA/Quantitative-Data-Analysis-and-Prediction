import numpy as np
import numba as nb
import pandas as pd
from sklearn.decomposition import PCA


@nb.njit(cache=True)
def _calc_pcs(X:np.array):
    # More or less similiar speed as sklearn.decomposition.PCA
    covariance_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    return eigen_values/sum(eigen_values)



def calc_composite_sig(instruments_dict, to_trade="THB_USD", col="close_sig", fit_window = ["2020-01-01","2020-12-31"]):
    # Merge all sigs into one df by timeframe
    instruments_dict_by_timeframe = {}
    temp_sig = {}
    temp_z= {}
    for timeframe in instruments_dict[to_trade].keys():
        # timeframe = 5Mins
        df = instruments_dict[to_trade][timeframe]
        df = df.add_prefix(f"{to_trade}_")
        for instrument, instrument_dict in instruments_dict.items():
            if instrument == to_trade:
                continue
            else:
                df=pd.merge(df,instrument_dict[timeframe].add_prefix(f"{instrument}_"),right_index=True, left_index=True, how="left")
        target = list(df.filter(regex=col).columns)[1:]
        print(f"{timeframe}\n----> {target}")
        instruments_dict_by_timeframe[timeframe] = df
        
        # PC CALCULATION
        df_fit = df[fit_window[0]:fit_window[1]].copy()
        df_fit_X = df_fit[target].dropna()

        pca=PCA()
        df_sig_pca = pca.fit_transform(df_fit_X)
        pcs = pca.explained_variance_ratio_
        print(f"--------> {pcs}")
        
        
        sig_composite = (df.loc[:,target]*pcs).sum(axis=1)
        sig_composite = sig_composite.to_frame("sig")
        z_composite = (df.loc[:,target]*pcs).sum(axis=1)
        z_composite = z_composite.to_frame("sig")
        temp_sig[timeframe]=sig_composite
        temp_z[timeframe] = z_composite
        
    instruments_dict["sig_composite"]=temp_sig
    instruments_dict["z_composite"]=temp_z
    
    
    return instruments_dict


    
    
    