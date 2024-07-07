import pandas as pd
import numpy as np
import math

def calculate_annualized_volatility_and_parkinson(instruments_dict, return_series=False, window=["2020-01-01","2023-12-31"]):
    """
    Calculate the annualized volatility and Parkinson number for each currency pair.

    Parameters:
    df (DataFrame): The dataframe containing the OHLCV data.
    freq (str): The frequency of the data ('1m', '5m', '30m', '1h', '4h', '1d').

    Returns:
    DataFrame: A dataframe containing the annualized volatility and Parkinson number for each currency pair.
    """
    vol_dict = {}
    parkinson_dict = {}
    GBM_parkinson_to_vol_ratio_dict = {}
    for instrument in instruments_dict:
        temp_vol = {}
        temp_parkinson = {}
        temp_GBM = {}
        for timeframe in instruments_dict[instrument]:
            df = instruments_dict[instrument][timeframe][window[0]:window[1]].copy()
            # Calculate the log returns
            np_log_return = np.log(df['close'] / df['close'].shift(1))

            # Convert the frequency to number of periods per year
            if timeframe == '1m':
                periods_per_year = 252 * 24 * 60
            elif timeframe == '2m':
                periods_per_year = 252 * 24 * 30
            elif timeframe == '3m':
                periods_per_year = 252 * 24 * 20
            elif timeframe == '4m':
                periods_per_year = 252 * 24 * 15
            elif timeframe == '5m':
                periods_per_year = 252 * 24 * 12
            elif timeframe == "15m":
                periods_per_year = 252 * 24 * 4
            elif timeframe == '30m':
                periods_per_year = 252 * 24 * 2
            elif timeframe == '1h':
                periods_per_year = 252 * 24
            elif timeframe == '2h':
                periods_per_year = 252 * 12
            elif timeframe == '3h':
                periods_per_year = 252 * 8
            elif timeframe == '4h':
                periods_per_year = 252 * 6
            elif timeframe == '8h':
                periods_per_year = 252 * 3
            elif timeframe == '12h':
                periods_per_year = 252 * 2
            elif timeframe == '1d':
                periods_per_year = 252

            # Calculate the rolling volatility
            np_vol = np_log_return.rolling(window=periods_per_year).std()


            # Calculate the annualized volatility
            annualized_vol = np_vol * np.sqrt(periods_per_year) * 100
            if return_series:
                temp_vol[timeframe] = annualized_vol
            else:
                temp_vol[timeframe] = np.round(annualized_vol.iloc[-1],2)

            # Calculate the Parkinson number
            rs = (np.log(df['high'] / df['low']) ** 2.0).rolling(window=periods_per_year).sum()
            annualized_parkinson_number = (rs / (4.0 * math.log(2.0) * periods_per_year)) ** 0.5 * np.sqrt(periods_per_year)*100
            if return_series:
                temp_parkinson[timeframe] = annualized_parkinson_number
            else:
                temp_parkinson[timeframe] = np.round(annualized_parkinson_number.iloc[-1],2)

            # GBM
            if return_series:
                temp_GBM[timeframe] = np.round(annualized_parkinson_number/annualized_vol,2)
            else:
                temp_GBM[timeframe] = np.round(annualized_parkinson_number.iloc[-1]/annualized_vol.iloc[-1],2)
        vol_dict[instrument] = temp_vol
        parkinson_dict[instrument] = temp_parkinson
        GBM_parkinson_to_vol_ratio_dict[instrument] = temp_GBM

    if return_series:

        # Loop over each instrument and timeframe
        temp = {}
        for i, output in enumerate([vol_dict, parkinson_dict,GBM_parkinson_to_vol_ratio_dict]):
            # Initialize an empty DataFrame
            df_combined = pd.DataFrame()
            for instrument in output:
                first_timeframe_flag = True
                for timeframe in output[instrument]:
                    # Convert the series to a DataFrame
                    df = output[instrument][timeframe].to_frame().add_suffix(f"_{timeframe}")
                    if first_timeframe_flag: 
                        df_combined = df.copy()
                        first_timeframe_flag = False
                        continue
                    # print(f"instrument: {instrument}, timeframe: {timeframe}, df: \n{df}\n\n")
                    # Add a column for the instrument and timeframe
                    # Append to the combined DataFrame
                    df_combined = pd.merge(df_combined, df, how='left', left_index=True, right_index=True)
                    # print(f"instrument: {instrument}, timeframe: {timeframe}, df_combined: \n{df_combined}\n\n")
            # Rename the columns
            
            temp[i] = df_combined
        return temp, vol_dict, parkinson_dict, GBM_parkinson_to_vol_ratio_dict
    else:
        return vol_dict, parkinson_dict,GBM_parkinson_to_vol_ratio_dict