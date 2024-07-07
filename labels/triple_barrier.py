import numpy as np
import pandas as pd
import numba as nb
from typing import Tuple, List

from tqdm import tqdm


# @nb.njit(cache=True) --> need to get rid of pandas series to be numbarised
def compute_triple_barrier_labels(
    price_series: pd.Series, 
    event_timestamps: pd.Series, 
    horizon_delta: int, 
    upper_delta: float=None, 
    lower_delta: float=None, 
    vol_span: int=20, 
    upper_z: float=None,
    lower_z: float=None,
    upper_label: int=1, 
    lower_label: int=-1,
    tf=4,
    labels = 2) -> Tuple[pd.Series, pd.Series]:
    assert labels in [2,3]
    """
    Calculate event labels according to the triple-barrier method. 
    
    price_series: 
    
    Return a series with both the original events and the labels. Labels 1, 0, 
    and -1 correspond to upper barrier breach, vertical barrier breach, and 
    lower barrier breach, respectively. 
    Also return series where the index is the start date of the label and the 
    values are the end dates of the label.
    """

    time_horizon_delta = horizon_delta*60*60*tf # to timestamp seconds if 4, then 60 *60
    series = pd.Series(np.log(price_series.values), index=event_timestamps)#price_series.index)

    
    n = len(price_series)
    np_labels = np.full(n, np.nan)
    np_label_dates = np.full(n, np.nan)
    
    if upper_z or lower_z:
        threshold = series.ewm(span=vol_span).std()
        threshold *= np.sqrt(horizon_delta / vol_span)

    for i,event_date in tqdm(enumerate(event_timestamps)):
        # print(i,event_date)
        date_barrier = event_date + time_horizon_delta

        start_price = series.loc[event_date]
        log_returns = series.loc[event_date:date_barrier] - start_price

        # First element of tuple is 1 or -1 indicating upper or lower barrier
        # Second element of tuple is first date when barrier was crossed
        candidates: List[Tuple[int, pd.Timestamp]] = list()

        # Add the first upper or lower date to candidates
        if upper_delta:
            _date = log_returns[log_returns > upper_delta].first_valid_index()
            if _date:
                candidates.append((upper_label, _date))
    
        if lower_delta:
            _date = log_returns[log_returns < lower_delta].first_valid_index()
            if _date:
                candidates.append((lower_label, _date))

        # Add the first upper_z and lower_z to candidates
        if upper_z:
            upper_barrier = upper_z * threshold[event_date]
            _date = log_returns[log_returns > upper_barrier].first_valid_index()
            if _date:
                candidates.append((upper_label, _date))

        if lower_z:
            lower_barrier = lower_z * threshold[event_date]
            _date = log_returns[log_returns < lower_barrier].first_valid_index()
            if _date:
                candidates.append((lower_label, _date))

        if candidates:
            # If any candidates, return label for first date
            label, label_date = min(candidates, key=lambda x: x[1])
        else:
            # If there were no candidates, time barrier was touched
            if labels == 3:
                label, label_date = 0, date_barrier
            elif labels == 2:
            # CHeck if return at this time_barrier is pos or neg then label -1/1 -> dont want 0
                end_returns = log_returns.iloc[-1]
                if end_returns >0:
                    label, label_date = upper_label, date_barrier
                else:
                    label, label_date = lower_label, date_barrier
        
        np_labels[i] = label
        np_label_dates[i] = label_date
        
    # Output

    event_spans_labels = pd.DataFrame({"t":event_timestamps, 't1':np_label_dates,'label': np_labels})
    return event_spans_labels

def calc_triple_barrier(df0,
                        col_series="1h_close",
                        col_timestamps = "1h_closeTime",
                        horizon_delta = 4,
                        vol_span = 10,
                        labels =2,
                        upper_z = 1.8,
                        upper_delta=None,
                        lower_z = -1.8,
                        lower_delta=None,
                        resample=None,
                        side = None,
                        fill_no_trades=True):
    if resample is not None:
        df=df0.resample(resample).last().copy()
        tf=int(resample[:-1])
    else:
        df=df0.copy()
        tf=1
    # timestamps =df.index
    
    event_spans_labels = compute_triple_barrier_labels(price_series = df[col_series],
                                                        event_timestamps = df[col_timestamps],
                                                        horizon_delta=horizon_delta,
                                                        vol_span = vol_span,
                                                        upper_delta=upper_delta,
                                                        lower_delta=lower_delta,
                                                        upper_z=upper_z,
                                                        lower_z=lower_z,
                                                        tf=tf,
                                                        labels=labels
                                                    )
    
    # Replace labels if aleady exisiting
    if "label" in df.columns:
        try:
            df.drop(columns=["label","t","t1"], inplace=True)
        except Exception as e:
            pass
    # Concatenate labels
    df = pd.merge(df0,event_spans_labels, right_index=True, left_index=True, how="left")
    if resample is not None:
        df["label"].fillna(method="bfill", inplace=True)
        
    if fill_no_trades and labels != 2:
        df["label"].replace(0, np.NaN, inplace=True)
        df["label"].fillna(method="bfill",inplace=True)
    
    if side is not None:
        try:
            # df.reset_index(inplace=True)
            # df.set_index("1h_closeTime", drop=False,inplace=True)
            # df["ret"] = np.log(df["1h_close"].shift(horizon_delta)/df["1h_close"])
            # df["ret"]=df["ret"].shift(-horizon_delta-1)
            # # np.log(df[col_series].loc[df['t1'].values].values) - np.log(df[col_series].loc[df['t1']])
            temp = df["label"] * df[side]  # meta-labeling
            # Label incorrect side as 0
            df["p_target"] = np.where(temp<= 0,0,1)
            # df.loc[df[f"meta_{side}"]<= 0, 'bin'] = 0
            # df.set_index("date_time",inplace=True)
        except Exception as e:
            # df.set_index("date_time",inplace=True)
            print(e)
            return df
        
        
    return df




# =============================================================================
# NUMBARISED TRIPLE BARRIER
# =============================================================================
# @nb.njit(cache=True) 
def first_hit(np_price, threshold):
    for idx, val in np.ndenumerate(np_price):
        if threshold>0:
            if val > threshold:
                return idx
        elif threshold <0:
            if val < threshold:
                return idx

# @nb.njit(cache=True) 
def compute_triple_barrier_labels1(
                                    np_price, 
                                    np_close_time, 
                                    horizon_delta_bars: int, 
                                    upper_delta: float=None, 
                                    lower_delta: float=None, 
                                    upper_label: int=1, 
                                    lower_label: int=-1,
                                    labels = 3,
                                    ):
    """
    Calculate event labels according to the triple-barrier method. 
    
    price_series: 
    
    Return a series with both the original events and the labels. Labels 1, 0, 
    and -1 correspond to upper barrier breach, vertical barrier breach, and 
    lower barrier breach, respectively. 
    Also return series where the index is the start date of the label and the 
    values are the end dates of the label.
    """

    horizon_delta_seconds = horizon_delta_bars*60 # since each bar is 5mins --> convert 300s to number of bars
    np_log_price = np.log(np_price)

    
    n = len(np_log_price)
    np_labels = np.full(n, np.nan)
    np_label_dates = np.full(n, np.nan)
    
    for i,t_seconds in tqdm(enumerate(np_close_time)): # t_seconds used to be event date
        # print(i,event_date)
        date_barrier = t_seconds + horizon_delta_seconds

        start_price = np_log_price[i]
        np_log_returns = np_log_price[i:i+horizon_delta_bars] - start_price
        np_close_time_t = np_close_time[i:i+horizon_delta_bars]
        # First element of tuple is 1 or -1 indicating upper or lower barrier
        # Second element of tuple is first date when barrier was crossed
        candidates: List[Tuple[int, int]] = list()

        # Add the first upper or lower date to candidates
        if upper_delta:
            _i = first_hit(np_log_returns, upper_delta)
            if _i is not None:
                hit_time = np_close_time_t[_i]
                candidates.append((upper_label, hit_time))
    
        if lower_delta:
            _i = first_hit(np_log_returns, lower_delta)
            if _i is not None:
                hit_time = np_close_time_t[_i]
                candidates.append((lower_label, hit_time))
            
        if len(candidates)>0:
            # If any candidates, return label for first date
            label, label_date = min(candidates, key=lambda x: x[1])
        else:
            if labels == 3:
            # If there were no candidates, time barrier was touched
                label, label_date = 0, date_barrier
            elif labels == 2:
                if np_log_returns[-1]<=0:
                    label, label_date = -1, date_barrier
                else:
                    label, label_date = 1, date_barrier

        # print(f"{i}-> {t_seconds}: {label} , {label_date}")
        np_labels[i] = label
        np_label_dates[i] = label_date
        
    # Output
    return np_close_time,np_label_dates,np_labels


def calc_triple_barrier1(df0,
                        col_series="1h_close",
                        col_timestamps = "1h_closeTime",
                        horizon_delta = 4,
                        upper_delta=2,
                        lower_delta=-2,
                        labels = 3
                        ):

    df=df0.copy()
    tf=1
    # timestamps =df.index
    if "label" in df.columns:
        df.drop(columns=["label","t","t1"], inplace=True)
    
    np_close_time,np_label_dates,np_labels = compute_triple_barrier_labels1(np_price = df[col_series].values,
                                                                            np_close_time = df[col_timestamps].values,
                                                                            horizon_delta_bars=horizon_delta,
                                                                            upper_delta=upper_delta,
                                                                            lower_delta=lower_delta,
                                                                            labels = labels
                                                                        )
    labels_df = pd.DataFrame({"t":np_close_time, 't1':np_label_dates,'label': np_labels})
    labels_df.index = pd.to_datetime(labels_df["t"], unit="s")
    labels_df.index.name = "date_time"
    # Replace labels if aleady exisiting

    # Concatenate labels
    df = pd.merge(df,labels_df, right_index=True, left_index=True, how="left")

        
        
    return df,labels_df



