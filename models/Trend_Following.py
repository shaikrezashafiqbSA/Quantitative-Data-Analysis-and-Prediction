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


def get_signal_qtl(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["L_pnl%"][i]> signals_dict["tide_long_TP1"][i]# and signals_dict["sig"][i-1]< signals_dict["L_lq"][i-1]
        elif side == "sell":
            signal = signals_dict["L_pnl%"][i]< signals_dict["tide_long_TP1"][i]
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["S_pnl%"][i]> signals_dict["tide_short_TP1"][i] #and signals_dict["sig"][i-1]> signals_dict["S_lq"][i-1]
        elif side == "sell":
            signal = signals_dict["S_pnl%"][i]< signals_dict["tide_short_TP1"][i]

    return signal

def get_signal_strengths(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["tide"][i] > 0 and signals_dict["tide"][i-1] < 0
            # signal = signals_dict["tide"][i] >=1
        elif side == "sell":
            tide_change = signals_dict["tide"][i] < 0 and signals_dict["tide"][i-1] > 0
            extreme_RR2_lb = signals_dict[f"tide_long_lb_RRRatio2"][i-1] > signals_dict["tide_long_RRRatio2"][i] and signals_dict["tide_long_RRRatio2"][i] > signals_dict["tide_long_RRRatio3"][i]
            extreme_RR2_ub = signals_dict[f"tide_long_ub_RRRatio2"][i-1] < signals_dict["tide_long_RRRatio2"][i]
            extreme_RR3_lb = signals_dict[f"tide_long_lb_RRRatio3"][i-1] > signals_dict["tide_long_RRRatio3"][i] and signals_dict["tide_long_RRRatio2"][i] > signals_dict["tide_long_RRRatio3"][i]
            extreme_RR3_ub = signals_dict[f"tide_long_ub_RRRatio3"][i-1] < signals_dict["tide_long_RRRatio3"][i]

            # SL1_hit = (np_closePx[i] < signals_dict["tide_long_SL1"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL1"][i-2]) 
            # TP1_hit = (np_closePx[i] > signals_dict["tide_long_TP1"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP1"][i-2])

            SL2_hit = (np_closePx[i] < signals_dict["tide_long_SL2"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL2"][i-2]) 
            TP2_hit = (np_closePx[i] > signals_dict["tide_long_TP2"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP2"][i-2])

            SL3_hit = (np_closePx[i] < signals_dict["tide_long_SL3"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL3"][i-2]) 
            TP3_hit = (np_closePx[i] > signals_dict["tide_long_TP3"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP3"][i-2])

            # signal = tide_change or (SL1_hit or TP1_hit) or (SL2_hit or TP2_hit) or (SL3_hit or TP3_hit)
            signal = tide_change or ((SL2_hit and extreme_RR2_ub) or (TP2_hit and not extreme_RR2_lb)) or ((SL3_hit and extreme_RR3_ub) or (TP3_hit and not extreme_RR3_lb))

        
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["tide"][i] < 0 and signals_dict["tide"][i-1] > 0
            #  signal = signals_dict["tide"][i] <=1
        elif side == "sell":
            tide_change = signals_dict["tide"][i] > 0 and signals_dict["tide"][i-1] < 0
            extreme_RR2_lb = signals_dict[f"tide_short_lb_RRRatio2"][i-1] > signals_dict["tide_short_RRRatio2"][i] and signals_dict["tide_short_RRRatio2"][i] > signals_dict["tide_short_RRRatio3"][i]
            extreme_RR2_ub = signals_dict[f"tide_short_ub_RRRatio2"][i-1] < signals_dict["tide_short_RRRatio2"][i]
            extreme_RR3_lb = signals_dict[f"tide_short_lb_RRRatio3"][i-1] > signals_dict["tide_short_RRRatio3"][i] and signals_dict["tide_short_RRRatio2"][i] < signals_dict["tide_short_RRRatio3"][i]
            extreme_RR3_ub = signals_dict[f"tide_short_ub_RRRatio3"][i-1] < signals_dict["tide_short_RRRatio3"][i]



            # SL1_hit = (np_closePx[i] > signals_dict["tide_short_SL1"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL1"][i-2]) 
            # TP1_hit = (np_closePx[i] < signals_dict["tide_short_TP1"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP1"][i-2])

            SL2_hit = (np_closePx[i] > signals_dict["tide_short_SL2"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL2"][i-2]) 
            TP2_hit = (np_closePx[i] < signals_dict["tide_short_TP2"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP2"][i-2])

            SL3_hit = (np_closePx[i] > signals_dict["tide_short_SL3"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL3"][i-2]) 
            TP3_hit = (np_closePx[i] < signals_dict["tide_short_TP3"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP3"][i-2])
            # signal = tide_change or (SL1_hit or TP1_hit) or (SL2_hit or TP2_hit) or (SL3_hit or TP3_hit)
            signal = tide_change or ((SL2_hit and extreme_RR2_ub) or (TP2_hit and not extreme_RR2_lb)) or ((SL3_hit and extreme_RR3_ub) or (TP3_hit and not extreme_RR3_lb))

    return signal

def get_signal_strengths_w_macros(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    
    if position == "long":        
        if side == "buy": 
            tide_change = signals_dict["tide"][i] > 0 and signals_dict["tide"][i-1] < 0
            macro_tide_change = (signals_dict["Y"][i] > 0 and signals_dict["Y"][i-1] < 0)
            signal = tide_change and macro_tide_change
        elif side == "sell":
            tide_change = signals_dict["tide"][i] < 0 and signals_dict["tide"][i-1] > 0
            extreme_RR2_lb = signals_dict[f"tide_long_lb_RRRatio2"][i-1] > signals_dict["tide_long_RRRatio2"][i] and signals_dict["tide_long_RRRatio2"][i] > signals_dict["tide_long_RRRatio3"][i]
            extreme_RR2_ub = signals_dict[f"tide_long_ub_RRRatio2"][i-1] < signals_dict["tide_long_RRRatio2"][i]
            extreme_RR3_lb = signals_dict[f"tide_long_lb_RRRatio3"][i-1] > signals_dict["tide_long_RRRatio3"][i] and signals_dict["tide_long_RRRatio2"][i] > signals_dict["tide_long_RRRatio3"][i]
            extreme_RR3_ub = signals_dict[f"tide_long_ub_RRRatio3"][i-1] < signals_dict["tide_long_RRRatio3"][i]

            # SL1_hit = (np_closePx[i] < signals_dict["tide_long_SL1"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL1"][i-2]) 
            # TP1_hit = (np_closePx[i] > signals_dict["tide_long_TP1"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP1"][i-2])

            SL2_hit = (np_closePx[i] < signals_dict["tide_long_SL2"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL2"][i-2]) 
            TP2_hit = (np_closePx[i] > signals_dict["tide_long_TP2"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP2"][i-2])

            SL3_hit = (np_closePx[i] < signals_dict["tide_long_SL3"][i-1]) and (np_closePx[i-1] > signals_dict["tide_long_SL3"][i-2]) 
            TP3_hit = (np_closePx[i] > signals_dict["tide_long_TP3"][i-1]) and (np_closePx[i-1] < signals_dict["tide_long_TP3"][i-2])

            # signal = tide_change or (SL1_hit or TP1_hit) or (SL2_hit or TP2_hit) or (SL3_hit or TP3_hit)
            signal = tide_change or ((SL2_hit and extreme_RR2_ub) or (TP2_hit and not extreme_RR2_lb)) or ((SL3_hit and extreme_RR3_ub) or (TP3_hit and not extreme_RR3_lb))

        
    elif position == "short":
        if side == "buy": 
            tide_change = (signals_dict["tide"][i] < 0 and signals_dict["tide"][i-1] > 0)
            macro_tide_change = (signals_dict["Y"][i] < 0 and signals_dict["Y"][i-1] > 0)
            signal = tide_change and macro_tide_change

        elif side == "sell":
            tide_change = signals_dict["tide"][i] > 0 and signals_dict["tide"][i-1] < 0
            extreme_RR2_lb = signals_dict[f"tide_short_lb_RRRatio2"][i-1] > signals_dict["tide_short_RRRatio2"][i] and signals_dict["tide_short_RRRatio2"][i] > signals_dict["tide_short_RRRatio3"][i]
            extreme_RR2_ub = signals_dict[f"tide_short_ub_RRRatio2"][i-1] < signals_dict["tide_short_RRRatio2"][i]
            extreme_RR3_lb = signals_dict[f"tide_short_lb_RRRatio3"][i-1] > signals_dict["tide_short_RRRatio3"][i] and signals_dict["tide_short_RRRatio2"][i] < signals_dict["tide_short_RRRatio3"][i]
            extreme_RR3_ub = signals_dict[f"tide_short_ub_RRRatio3"][i-1] < signals_dict["tide_short_RRRatio3"][i]



            # SL1_hit = (np_closePx[i] > signals_dict["tide_short_SL1"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL1"][i-2]) 
            # TP1_hit = (np_closePx[i] < signals_dict["tide_short_TP1"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP1"][i-2])

            SL2_hit = (np_closePx[i] > signals_dict["tide_short_SL2"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL2"][i-2]) 
            TP2_hit = (np_closePx[i] < signals_dict["tide_short_TP2"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP2"][i-2])

            SL3_hit = (np_closePx[i] > signals_dict["tide_short_SL3"][i-1]) and (np_closePx[i-1] < signals_dict["tide_short_SL3"][i-2]) 
            TP3_hit = (np_closePx[i] < signals_dict["tide_short_TP3"][i-1]) and (np_closePx[i-1] > signals_dict["tide_short_TP3"][i-2])
            # signal = tide_change or (SL1_hit or TP1_hit) or (SL2_hit or TP2_hit) or (SL3_hit or TP3_hit)
            signal = tide_change or ((SL2_hit and extreme_RR2_ub) or (TP2_hit and not extreme_RR2_lb)) or ((SL3_hit and extreme_RR3_ub) or (TP3_hit and not extreme_RR3_lb))

    return signal

def get_signal_default(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",entry_i = None)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    sig_lagged = i -sig_lag
    if sig_lagged >= i-1:
        sig_lagged = i
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["sig"][i] > 0
        elif side == "sell":
            signal = signals_dict["sig"][i] == 0
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["sig"][i] == 0
        elif side == "sell":
            signal = signals_dict["sig"][i] > 0 

    return signal


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
#                                                           tide signals
# =========================================================================================================================================
def get_tide_sig(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:

    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    i_lagged = i -sig_lag
    if i_lagged >= len(signals_dict["tide"]):
        i_lagged = i
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["tide"][i_lagged] == 1 and signals_dict["tide"][i_lagged-1] 
        elif side == "sell":
            signal = signals_dict["tide"][i_lagged] < 1
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["tide"][i_lagged] == -1 and signals_dict["tide"][i_lagged-1] != 1
        elif side == "sell":
            signal = signals_dict["tide"][i_lagged] == 0
    # if signal:
    #     print(f"i: {i}, signal: {signal}")
    return signal


def get_tide_sig_TP(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:

    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    i_lagged = i -sig_lag
    if i_lagged >= len(signals_dict["tide"]):
        i_lagged = i
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["tide"][i_lagged] > 0
        elif side == "sell":
            signal = signals_dict["tide"][i_lagged] <= 0
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["tide"][i_lagged] <= 0
        elif side == "sell":
            signal = signals_dict["tide"][i_lagged] > 0 
    # if signal:
    #     print(f"i: {i}, signal: {signal}")
    return signal

# =========================================================================================================================================
#                                                           z signals
# =========================================================================================================================================
# def get_z_sig(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:
    
#     # if i < L_q_lookback or i < S_q_lookback:
#     #     return False
#     # print(signals_dict)
#     L_buy=kwargs.get("L_buy",1)
#     L_sell=kwargs.get("L_sell",1)
#     S_buy=kwargs.get("S_buy",-1)
#     S_sell=kwargs.get("S_sell",-1)

#     # print(f"S_sell: {kwargs.get('S_sell',-1)}")
#     sig_lagged = i -sig_lag
#     if sig_lagged >= i-1:
#         sig_lagged = i
#     if position == "long":        
#         if side == "buy": 
#             signal = signals_dict["sig"][sig_lagged] >=L_buy
#         elif side == "sell":
#             signal = signals_dict["sig"][sig_lagged] <L_sell
            
#     elif position == "short":
#         if side == "buy": 
#             signal = signals_dict["sig"][sig_lagged] <= S_buy
#         elif side == "sell":
#             signal = signals_dict["sig"][sig_lagged] > S_sell

#     return signal


def get_z_sig(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:
    
    # if i < L_q_lookback or i < S_q_lookback:
    #     return False
    # print(signals_dict)
    L_buy=kwargs.get("L_buy",1)
    L_sell=kwargs.get("L_sell",1)
    S_buy=kwargs.get("S_buy",-1)
    S_sell=kwargs.get("S_sell",-1)

    # print(f"S_sell: {kwargs.get('S_sell',-1)}")
    sig_lagged = i -sig_lag
    if sig_lagged >= i-1:
        sig_lagged = i
    # if position == "long":        
    #     if side == "buy": 
    #         signal = signals_dict["sig_tide"][sig_lagged] >=L_buy
    #     elif side == "sell":
    #         signal = signals_dict["sig_tide"][sig_lagged] <L_sell
            
    # elif position == "short":
    #     if side == "buy": 
    #         signal = signals_dict["sig_tide"][sig_lagged] <= S_buy
    #     elif side == "sell":
    #         signal = signals_dict["sig_tide"][sig_lagged] > S_sell
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["sig_tide"][sig_lagged] <= L_buy
        elif side == "sell":
            signal = signals_dict["sig_tide"][sig_lagged] > L_sell
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["sig_tide"][sig_lagged] >= S_buy
        elif side == "sell":
            signal = signals_dict["sig_tide"][sig_lagged] < S_sell

    return signal

def get_z_sig_TP(i,np_closePx, signals_dict, sig_lag=0, position="long",side="buy",**kwargs)-> bool:
    L_buy=kwargs.get("L_buy",1)
    L_sell=kwargs.get("L_sell",1)
    S_buy=kwargs.get("S_buy",-1)
    S_sell=kwargs.get("S_sell",-1)

    long_closeIdx = kwargs.get("long_closeIdx",None)
    short_closeIdx = kwargs.get("short_closeIdx",None)
    # This kwargs.get does: if kwargs.get("L_buy",1) exists, then use that value, else use 1
    sig_lagged = i -sig_lag
    if sig_lagged >= i-1:
        sig_lagged = i
    
    if position == "long":
        profitable_RR = signals_dict[f"sig_S_RR1"][i] < 1 # Risk > reward disproportionately. 
            
        if side == "buy": 
            time_since_trade_int = i - long_closeIdx
        
            sig_change = signals_dict["sig_tide"][sig_lagged] <=L_buy
            # macro_tide_change = (signals_dict["Y"][i] > 0 and signals_dict["Y"][i-1] < 0)
            signal = sig_change and profitable_RR 

        elif side == "sell":
            entry_i = kwargs.get("entry_i",None)
            time_in_trade_int = i - entry_i

            TP1_hit = (np_closePx[i] > signals_dict["sig_L_TP1"][i-1]) and (np_closePx[i-1] < signals_dict["sig_L_TP1"][i-2])
            SL1_hit = (np_closePx[i] < signals_dict["sig_L_SL1"][i-1]) and (np_closePx[i-1] > signals_dict["sig_L_SL1"][i-2]) 
            TP1_t_hit = signals_dict["sig_L_TP1_t"][i] > time_in_trade_int
            SL1_t_hit = signals_dict["sig_L_SL1_t"][i] > time_in_trade_int

            TP2_hit = (np_closePx[i] > signals_dict["sig_L_TP2"][i-1]) and (np_closePx[i-1] < signals_dict["sig_L_TP2"][i-2])
            SL2_hit = (np_closePx[i] < signals_dict["sig_L_SL2"][i-1]) and (np_closePx[i-1] > signals_dict["sig_L_SL2"][i-2]) 
            TP2_t_hit = signals_dict["sig_L_TP2_t"][i] > time_in_trade_int
            SL2_t_hit = signals_dict["sig_L_SL2_t"][i] > time_in_trade_int

            TP3_hit = (np_closePx[i] > signals_dict["sig_L_TP3"][i-1]) and (np_closePx[i-1] < signals_dict["sig_L_TP3"][i-2])
            SL3_hit = (np_closePx[i] < signals_dict["sig_L_SL3"][i-1]) and (np_closePx[i-1] > signals_dict["sig_L_SL3"][i-2]) 
            TP3_t_hit = signals_dict["sig_L_TP3_t"][i] > time_in_trade_int
            SL3_t_hit = signals_dict["sig_L_SL3_t"][i] > time_in_trade_int


            sig_change = signals_dict["sig_tide"][sig_lagged] > L_sell

            # SL1_exit = SL1_hit and signals_dict["sig_long_SL1_t"][i]
            # signal = sig_change and profitable_RR
            # signal = sig_change or ((SL2_hit and profitable_RR) or (TP2_hit and not profitable_RR)) or ((SL3_hit and profitable_RR) or (TP3_hit and not profitable_RR))
            TPs_hit = profitable_RR or (TP1_hit or TP2_hit or TP3_hit) 
            TPs_t_hit = profitable_RR or (TP1_t_hit or TP2_t_hit or TP3_t_hit)
            SLs_hit = not profitable_RR or (SL1_hit or SL2_hit or SL3_hit)
            SLs_t_hit = not profitable_RR or (SL1_t_hit or SL2_t_hit or SL3_t_hit)
            # signal = TPs_hit or TPs_t_hit or SLs_hit or SLs_t_hit   
            signal = TPs_hit or SLs_hit   
        
    elif position == "short":
        profitable_RR = signals_dict[f"sig_L_RR1"][i] < 1 # Risk > reward disproportionately. 
    
        if side == "buy": 
            time_since_trade_int = i - short_closeIdx
        
            sig_change = signals_dict["sig_tide"][sig_lagged] >= S_buy

            # macro_tide_change = (signals_dict["Y"][i] < 0 and signals_dict["Y"][i-1] > 0)
            signal = sig_change and profitable_RR # and time_since_trade_int > profitable_RR*signals_dict["sig_long_SL3_t"][i]

        elif side == "sell":
            entry_i = kwargs.get("entry_i",None)
            time_in_trade_int = i - entry_i

            TP1_hit = (np_closePx[i] < signals_dict["sig_S_TP1"][i-1]) and (np_closePx[i-1] > signals_dict["sig_S_TP1"][i-2])
            SL1_hit = (np_closePx[i] > signals_dict["sig_S_SL1"][i-1]) and (np_closePx[i-1] < signals_dict["sig_S_SL1"][i-2]) 
            TP1_t_hit = signals_dict["sig_S_TP1_t"][i] > time_in_trade_int
            SL1_t_hit = signals_dict["sig_S_SL1_t"][i] > time_in_trade_int

            TP2_hit = (np_closePx[i] < signals_dict["sig_S_TP2"][i-1]) and (np_closePx[i-1] > signals_dict["sig_S_TP2"][i-2])
            SL2_hit = (np_closePx[i] > signals_dict["sig_S_SL2"][i-1]) and (np_closePx[i-1] < signals_dict["sig_S_SL2"][i-2]) 
            TP2_t_hit = signals_dict["sig_S_TP2_t"][i] > time_in_trade_int
            SL2_t_hit = signals_dict["sig_S_SL2_t"][i] > time_in_trade_int

            TP3_hit = (np_closePx[i] < signals_dict["sig_S_TP3"][i-1]) and (np_closePx[i-1] > signals_dict["sig_S_TP3"][i-2])
            SL3_hit = (np_closePx[i] > signals_dict["sig_S_SL3"][i-1]) and (np_closePx[i-1] < signals_dict["sig_S_SL3"][i-2]) 
            TP3_t_hit = signals_dict["sig_S_TP3_t"][i] > time_in_trade_int
            SL3_t_hit = signals_dict["sig_S_SL3_t"][i] > time_in_trade_int

            sig_change = signals_dict["sig_tide"][sig_lagged] < S_sell
            
            # SL1_exit = SL1_hit and signals_dict["sig_long_SL1_t"][i]
            # signal = sig_change and profitable_RR 
            # signal = sig_change or ((SL2_hit) or (TP2_hit and profitable_RR)) or ((SL3_hit) or (TP3_hit and profitable_RR))
            TPs_hit = profitable_RR or (TP1_hit or TP2_hit or TP3_hit) 
            TPs_t_hit = profitable_RR or (TP1_t_hit or TP2_t_hit or TP3_t_hit)
            SLs_hit = not profitable_RR or (SL1_hit or SL2_hit or SL3_hit)
            SLs_t_hit = not profitable_RR or (SL1_t_hit or SL2_t_hit or SL3_t_hit)
            # signal = TPs_hit or TPs_t_hit or SLs_hit or SLs_t_hit   
            signal = TPs_hit or SLs_hit   
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
                
    
        if signal_function is None:
            _get_signal = get_signal
        elif signal_function == "qtl":
            _get_signal = get_signal_qtl
        elif signal_function == "strengths":
            _get_signal = get_signal_strengths
        elif signal_function == "strength_w_macros":
            _get_signal = get_signal_strengths_w_macros
        elif signal_function == "tide":
            _get_signal = get_tide_sig
        elif signal_function == "tide_TP":
            _get_signal = get_tide_sig_TP
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