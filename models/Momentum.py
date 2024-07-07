import numpy as np

  
def get_signal(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
            
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["label"][i-sig_lag] == 1
        elif side == "sell":
            signal = signals_dict["label"][i-sig_lag] != 1
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["label"][i-sig_lag] == -1
        elif side == "sell":
            signal = signals_dict["label"][i-sig_lag] != -1

    return signal



def get_signal_default(i, np_closePx, signals_dict,sig_lag=0, position="long",side="buy", **kwargs)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    L_buy=kwargs.get("L_buy",0)
    L_sell=kwargs.get("L_sell",0)
    S_buy=kwargs.get("S_buy",-3)
    S_sell=kwargs.get("S_sell",3)
    try:
        if position == "long":        
            if side == "buy": 
                signal = signals_dict["sig"][i-sig_lag] >=L_buy
            elif side == "sell":
                signal = signals_dict["sig"][i-sig_lag] <=L_sell
                
        elif position == "short":
            if side == "buy": 
                signal = signals_dict["sig"][i-sig_lag] <=S_buy
            elif side == "sell":
                signal = signals_dict["sig"][i-sig_lag] >=S_sell

        return signal
    except Exception as e:
        return False


def get_signal_Y(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
            
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["Y"][i-sig_lag]> 0
        elif side == "sell":
            signal = signals_dict["Y"][i-sig_lag]< 0
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["Y"][i-sig_lag]< 0
        elif side == "sell":
            signal = signals_dict["Y"][i-sig_lag]> 0

    return signal
    
# =========================================================================================================================================
#                                                           z signals
# =========================================================================================================================================

def get_z_sig(i, np_closePx, signals_dict, sig_lag=0, position="long", side="buy", **kwargs)-> bool:
    L_buy = kwargs.get("L_buy",1)
    L_sell = kwargs.get("L_sell",1)
    S_buy = kwargs.get("S_buy",-1)
    S_sell = kwargs.get("S_sell",-1)

    i_lagged = i - sig_lag
    if i_lagged < 0: 
        return False
    elif i_lagged >= len(np_closePx):
        return False

    signal = False
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["zscore"][i_lagged] >= L_buy
        elif side == "sell":
            signal = signals_dict["zscore"][i_lagged] < L_sell
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["zscore"][i_lagged] <= S_buy
        elif side == "sell":
            signal = signals_dict["zscore"][i_lagged] > S_sell

    return signal


def get_z_sig_TP(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:
    L_buy=kwargs.get("L_buy",1)
    L_sell=kwargs.get("L_sell",1)
    S_buy=kwargs.get("S_buy",-1)
    S_sell=kwargs.get("S_sell",-1)

    long_closeIdx = kwargs.get("long_closeIdx",None)
    short_closeIdx = kwargs.get("short_closeIdx",None)
    
    long_openIdx = kwargs.get("long_closeIdx",None)
    short_openIdx = kwargs.get("short_closeIdx",None)

    i_lagged = i -sig_lag
    if i_lagged > i:
        i_lagged = i
    elif i_lagged < 0:
        return False
    
    L_buy_signal_change = signals_dict["zscore"][i_lagged] >= L_buy # -1
    L_buy_profitable_RR = signals_dict["L_RR1"][i_lagged] <= 1
    L_buy_signal = L_buy_signal_change and L_buy_profitable_RR 

    S_buy_signal_change = signals_dict["zscore"][i_lagged] <= S_buy
    S_buy_profitable_RR = signals_dict["S_RR1"][i_lagged] <= 1
    S_buy_signal = S_buy_signal_change and S_buy_profitable_RR 
    
    if position == "long":
        signal = False
        if side == "buy": 
            # L_buy_signal_change = signals_dict["zscore"][i_lagged] <= L_buy # -1
            # L_buy_profitable_RR = signals_dict["L_RR1"][i_lagged] <= 1
            # L_buy_signal = L_buy_signal_change and L_buy_profitable_RR 
            
            signal = L_buy_signal
        elif side == "sell":
            signal_change = signals_dict["zscore"][i_lagged] < L_sell
            profitable_RR = signals_dict["S_RR1"][i_lagged] < 1
            long_openIdx = kwargs.get("long_openIdx",None)

            TP1_hit = (signals_dict["high"][i] > signals_dict["L_TP1"][long_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["L_TP1"][long_openIdx])
            TP2_hit = (signals_dict["high"][i] > signals_dict["L_TP2"][long_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["L_TP2"][long_openIdx])
            TP3_hit = (signals_dict["high"][i] > signals_dict["L_TP3"][long_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["L_TP3"][long_openIdx])
            SL1_hit = (signals_dict["low"][i] < signals_dict["L_SL1"][long_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["L_SL1"][long_openIdx])
            SL2_hit = (signals_dict["low"][i] < signals_dict["L_SL2"][long_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["L_SL2"][long_openIdx])
            SL3_hit = (signals_dict["low"][i] < signals_dict["L_SL3"][long_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["L_SL3"][long_openIdx])

            TP1_hit_t = i-long_openIdx > signals_dict["L_TP1_t"][long_openIdx]
            TP2_hit_t = i-long_openIdx > signals_dict["L_TP2_t"][long_openIdx]
            TP3_hit_t = i-long_openIdx > signals_dict["L_TP3_t"][long_openIdx]
            SL1_hit_t = i-long_openIdx > signals_dict["L_SL1_t"][long_openIdx]
            SL2_hit_t = i-long_openIdx > signals_dict["L_SL2_t"][long_openIdx]
            SL3_hit_t = i-long_openIdx > signals_dict["L_SL3_t"][long_openIdx]

            # TPs_hit = (TP1_hit or TP2_hit or TP3_hit) # THIS WORKS
            # SLs_hit = (SL1_hit or SL2_hit or SL3_hit) # THIS WORKS

            # TPs_hit = (TP1_hit_t or TP2_hit_t or TP3_hit_t) or (TP1_hit or TP2_hit or TP3_hit) 
            # SLs_hit = (SL1_hit_t or SL2_hit_t or SL3_hit_t) or (SL1_hit or SL2_hit or SL3_hit) 
            
            TPs_hit = (TP1_hit_t and TP1_hit) or (TP2_hit_t and TP2_hit) or (TP3_hit_t and TP3_hit) 
            SLs_hit = (SL1_hit_t and SL1_hit) or (SL2_hit_t and SL2_hit) or (SL3_hit_t and SL3_hit) 

            # signal = (TPs_hit or SLs_hit) and signal_change # THIS WORKS
            signal = ((TPs_hit or SLs_hit) and signal_change) or S_buy_signal # THIS WORKS
            # signal = (TPs_hit or SLs_hit) and profitable_RR

    elif position == "short":
        signal = False
        if side == "buy": 
            # S_buy_signal_change = signals_dict["zscore"][i_lagged] >= S_buy
            # S_buy_profitable_RR = signals_dict["S_RR1"][i_lagged] <= 1
            # S_buy_signal = S_buy_signal_change and S_buy_profitable_RR 

            signal = S_buy_signal
        elif side == "sell":
            signal_change = signals_dict["zscore"][i_lagged] > S_sell
            profitable_RR = signals_dict["S_RR1"][i_lagged] < 1
            short_openIdx = kwargs.get("short_openIdx",None)

            TP1_hit = (signals_dict["low"][i] < signals_dict["S_TP1"][short_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["S_TP1"][short_openIdx])
            TP2_hit = (signals_dict["low"][i] < signals_dict["S_TP2"][short_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["S_TP2"][short_openIdx])
            TP3_hit = (signals_dict["low"][i] < signals_dict["S_TP3"][short_openIdx]) #and (signals_dict["low"][i-1] > signals_dict["S_TP3"][short_openIdx])
            SL1_hit = (signals_dict["high"][i] > signals_dict["S_SL1"][short_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["S_SL1"][short_openIdx]) 
            SL2_hit = (signals_dict["high"][i] > signals_dict["S_SL2"][short_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["S_SL2"][short_openIdx]) 
            SL3_hit = (signals_dict["high"][i] > signals_dict["S_SL3"][short_openIdx]) #and (signals_dict["high"][i-1] < signals_dict["S_SL3"][short_openIdx]) 
            
            TP1_hit_t = i-short_openIdx > signals_dict["S_TP1_t"][short_openIdx]
            TP2_hit_t = i-short_openIdx > signals_dict["S_TP2_t"][short_openIdx]
            TP3_hit_t = i-short_openIdx > signals_dict["S_TP3_t"][short_openIdx]
            SL1_hit_t = i-short_openIdx > signals_dict["S_SL1_t"][short_openIdx]
            SL2_hit_t = i-short_openIdx > signals_dict["S_SL2_t"][short_openIdx]
            SL3_hit_t = i-short_openIdx > signals_dict["S_SL3_t"][short_openIdx]

            # TPs_hit = (TP1_hit or TP2_hit or TP3_hit) # THIS WORKS
            # SLs_hit = (SL1_hit or SL2_hit or SL3_hit) # THIS WORKS

            # TPs_hit = (TP1_hit_t or TP2_hit_t or TP3_hit_t) or (TP1_hit or TP2_hit or TP3_hit) 
            # SLs_hit = (SL1_hit_t or SL2_hit_t or SL3_hit_t) or (SL1_hit or SL2_hit or SL3_hit) 

            TPs_hit = (TP1_hit_t and TP1_hit) or (TP2_hit_t and TP2_hit) or (TP3_hit_t and TP3_hit) 
            SLs_hit = (SL1_hit_t and SL1_hit) or (SL2_hit_t and SL2_hit) or (SL3_hit_t and SL3_hit) 

            # signal = (TPs_hit or SLs_hit) and signal_change # THIS WORKS
            signal = ((TPs_hit or SLs_hit) and signal_change) or L_buy_signal # THIS WORKS
            # signal = (TPs_hit or SLs_hit) and profitable_RR
            
    return signal



class model:
    def run_model(df, model_config, model_state,position_sizing_to_trade=True, signals = ["sig", "L_uq", "L_lq", "S_uq", "S_lq"]):
        """
        Runs model
        model_state to be updated
        
        """
        #TODO: MIGRATE ALL THESE OUTSIDE METHOD. dont need to initialise everytime
        signal_function = model_config["signals"]["signal_function"]
        kline_to_trade = model_config["model_instruments"]["kline_to_trade"]
        long_trade_notional = model_config["equity_settings"]["long_trade_notional"]
        short_trade_notional = model_config["equity_settings"]["short_trade_notional"]
        
        # signals related
        signals_dict = {}
        if signals is not None:
            for signal in signals:
                signals_dict[signal] = df[signal].values
                
    
        if (signal_function is None) or (signal_function == "label"):
            _get_signal = get_signal
        elif signal_function == "z_sig":
            _get_signal = model.get_z_sig
        elif signal_function == "z_sig_TP":
            _get_signal = model.get_z_sig_TP
        elif signal_function == "default":
            _get_signal = get_signal_default
        else:
            raise Exception("Not valid signal function")
            
        # LIMITATION: IF df is trimmed to save on ram -> this i will be different
        # index i for last row of df
        i = len(df)-1
        
        
        np_closePx = df[kline_to_trade].values
        np_tradable = df["tradable"].values
        np_session_closing = df["session_closing"].values
        
        tradable = np_tradable[i]
        session_closing = np_session_closing[i]
        
        if position_sizing_to_trade:
            np_position_sizing = np.full(len(df), 1)
        else:
            np_position_sizing = df['size'].values
        
        # =============================================================================
        # ENTRIES
        # =============================================================================
    
        # ---------- #
        # ENTER LONG
        # ---------- #
        # TODO: intrabar position flipping --> remove 2nd and not. 
        if (not model_state['long']["in_position"] and not model_state['short']["in_position"]) and tradable and not session_closing and model_config['equity_settings']["long_trade_notional"]>0:
            signal = _get_signal(i,np_closePx,signals_dict, sig_lag=0, position="long",side="buy")
            if signal:
                model_state["long"]["buy_signal"] = True
                model_state["long"]["buy_quantity_expected"] = np.round(long_trade_notional * np_position_sizing[i],2)
                model_state["long"]["buy_price_expected"] = np_closePx[i]
                model_state["long"]["signal_event"] = True
                return model_state
            
        # ---------- #
        # ENTER SHORT
        # ---------- #
        if (not model_state['short']["in_position"] and not model_state['long']["in_position"]) and tradable and not session_closing and model_config['equity_settings']["short_trade_notional"]>0:
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=0, position="short",side="buy")
            if signal: 
                model_state["short"]["sell_signal"] = True
                model_state["short"]["sell_quantity_expected"] = np.round(short_trade_notional * np_position_sizing[i],2)
                model_state["short"]["sell_price_expected"] = np_closePx[i]
                model_state["short"]["signal_event"] = True
                return model_state
            
            
        # =============================================================================
        # EXITS
        # =============================================================================     
        
        # ========== #
        # LONG
        # ========== #    
        if model_state['long']["in_position"]:# and (i > model_state["long"]["open_index"]): 
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=0, position="long",side="sell")
            
            # ---------- #
            # EXIT LONG
            # ---------- #   
            signal_triggered_flag = (signal and tradable and not session_closing)
            tradable_but_session_closing =  (tradable and session_closing)
    
                    
            if (signal_triggered_flag or tradable_but_session_closing): 
                model_state["long"]["sell_signal"] = True
                model_state["long"]["sell_price_expected"] = np_closePx[i]
                # model_state["short"]["sell_quantity_expected"] = model_state["short"]["buy_quantity_actual"]
                model_state["long"]["signal_event"] = True
                return model_state
            
            
        # ========== #
        # SHORT
        # ========== #  
        if model_state['short']["in_position"]:#  and (i > model_state["short"]["open_index"]):
            signal = _get_signal(i,np_closePx, signals_dict, sig_lag=0, position="short",side="sell")
            
            # ---------- #
            # EXIT SHORT
            # ---------- #
            signal_triggered_flag = (signal and tradable and not session_closing)
            tradable_but_session_closing = (tradable and session_closing)
     
                
            if (signal_triggered_flag or tradable_but_session_closing): 
                model_state["short"]["buy_signal"] = True
                model_state["short"]["buy_price_expected"]  = np_closePx[i]
                # model_state["short"]["buy_quantity_expected"] = model_state["short"]["sell_quantity_actual"]
                model_state["short"]["signal_event"] = True
                return model_state
            
            
            
        return model_state  


# if __name__ == "__main__":
#     #%%
#     # from models import Mean_Reversion
#     # model_state2 = Mean_Reversion.run_mean_reversion_model(df, model_config, model_state)
#     pass