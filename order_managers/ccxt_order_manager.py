import requests
import json
import time
import warnings
import pandas as pd
import numpy as np
import copy
from datetime import datetime as dt

from order_manager.order_resp_default import order_resp_default
from logger.logger import logger
from data_manager import db_manager
from order_manager.model_state_default import model_state_default

class CCXT_Client():
    """
    bearer token: 
        eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiUGV0ZXIgV3UgV2VpIiwiaWF0IjoxNjY5NzA3ODcxfQ.iPP0jkN9xVjqFSNHZxhkE6zeKhnuPOwxv61LvxhKRi8
    
    """
    def __init__(self,model_config, url: str="http://18.140.28.101:8081/api/"):
        self.url = url
        self.auth_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiUGV0ZXIgV3UgV2VpIiwiaWF0IjoxNjY5NzA3ODcxfQ.iPP0jkN9xVjqFSNHZxhkE6zeKhnuPOwxv61LvxhKRi8'
        self.headers = {'Authorization': 'Bearer '+ self.auth_token}
        self.model_config = model_config
        
        
    
        
    def create_market_order(self, symbol:str, side:str, quantity:float) -> dict:
        if side == "buy":
            order_dict = {"from":symbol[:3].upper(), 
                          "to":symbol[-3:].upper(),
                          "action": side.lower(),
                          "qty": quantity}
        elif side == "sell":
            order_dict = {"from":symbol[-3:].upper(), 
                          "to":symbol[:3].upper(),
                          "action": side.lower(),
                          "qty": quantity}
        t1 = time.time_ns()
        resp = requests.post(url=self.url + "order/create", json=order_dict, headers=self.headers)
        print(f"{t1} - {resp}")
        return resp,t1
    
    def create_limit_order(self, symbol:str, side:str, price:float, quantity:float, time_in_force = "GTC") -> dict:
        if side == "buy":
            order_dict = {"from":symbol[:3].upper(), 
                          "to":symbol[-3:].upper(),
                          "action": side.lower(),
                          "qty": quantity,
                          "order_type": "Limit",
                          "time_in_force": time_in_force,
                          "price":price}
        elif side == "sell":
            order_dict = {"from":symbol[-3:].upper(), 
                          "to":symbol[:3].upper(),
                          "action": side.lower(),
                          "qty": quantity,
                          "order_type": "Limit",
                          "time_in_force": time_in_force,
                          "price":price}
            
        t1 = time.time_ns()
        resp = requests.post(url=self.url + "order/create", json=order_dict, headers=self.headers)
        print(f"{t1} - {resp}")
        return resp,t1

    
    def cancel_order(self, symbol, side, quantity, order_id):
        if side == "buy":
            order_dict = {"from":symbol[:3].upper(), 
                          "to":symbol[-3:].upper(),
                          "action": side.lower(),
                          "qty": quantity,
                          "order_id": str(order_id)}
        elif side == "sell":
            order_dict = {"from":symbol[-3:].upper(), 
                          "to":symbol[:3].upper(),
                          "action": side.lower(),
                          "qty": quantity,
                          "order_id": str(order_id)}
        t1 = time.time_ns()
        # try:
        resp = requests.post(url=self.url + "order/cancel", json=order_dict, headers=self.headers)
        # except Exception as e:
            
        if resp.status_code != 200:
            raise Exception(f"Order cancellation failed for {symbol} - {side} - {quantity} - {order_id}")
            
        print(f"{t1} - {resp}")
        return resp,t1

    def cancel_order_model_state_update(self, model_state, position, side):
        if model_state[position][f"{side}_pending"]:
            quantity_to_cancel = model_state[position][f"{side}_quantity_expected"]
            order_id_to_cancel = str(model_state[position]["order_id"])
            logger.info(f"Cancelling order for {position} | {side}_pending | {side}_quantity_expected: {quantity_to_cancel} | order_id: {order_id_to_cancel}")
            
            # Try to cancel orders, if failed (i.e; order has already been filled then continue)

            cancel_order_resp, cancel_timestamp = self.cancel_order(symbol=self.model_config["model_instruments"]["instrument_to_trade"], 
                                                                    side = side,
                                                                    quantity=quantity_to_cancel,
                                                                    order_id = order_id_to_cancel)
            # THIS ASSUMES order can has been immediately cancelled
            model_state[position][f"{side}_pending"] = False
            model_state[position][f"{side}_quantity_expected"] = 0
            model_state[position][f"{side}_price_expected"] = 0
            model_state[position]["order_id"] = 0

        return model_state
                
    def get_order_executions(self):
        order_executions_df = db_manager.read_table(table_name="order_executions")
        order_executions_df["cl_order_id"] = order_executions_df["cl_order_id"].values.astype(np.int64)
        order_executions_df = order_executions_df.sort_values(by=["cl_order_id", "created_at"])
        return order_executions_df
    
    def get_order_status_using_cl_order_id(self, symbol, cl_order_id):
        order_executions_df = self.get_order_executions()
        order_executions_df = order_executions_df[order_executions_df["cl_order_id"]==cl_order_id]
        
        return order_executions_df
    
    def get_order_status(self, symbol, order_request_timestamp) -> dict:
        symbol_sep_with_slash = symbol[:3]+"/" +symbol[-3:]
        collected_updated_orders = False
        i=0
        while not collected_updated_orders and i < 3:
            # order_executions_df = db_manager.read_table(table_name="order_executions")
            order_executions_df = self.get_order_executions()
            order_executions_df["executed-requested_ms"] = order_executions_df['created_at'].values.astype(np.int64) - order_request_timestamp
            
            # Select only specified symbol orders
            order_executions_df = order_executions_df[order_executions_df["symbol"]==symbol_sep_with_slash]
            order_executions_df=order_executions_df.sort_values(by="cl_order_id")
            # If limit order this shouldnt search for only filled orders
            # filled_orders = order_executions_df[order_executions_df['execution_status']=="F"]
            # order_resp = filled_orders[filled_orders['executed-requested_ms']>0.0]
    
            order_resp = order_executions_df[order_executions_df['executed-requested_ms']>0.0]
    
    
            if len(order_resp) == 1:
                collected_updated_orders  = True
                return order_resp, order_executions_df
            elif len(order_resp) > 1:
                order_resp = order_resp#.iloc[-1,:]
                collected_updated_orders  = True
                return order_resp, order_executions_df
            else:
                i+=1
                print(f"FAILED TO GET order_resp... retrying in 1s (retry count: {i})")
                time.sleep(1)
                collected_updated_orders = False
                # QUERY Order_executions table again
        else:
            # print order resp success
            logger.error(f"ORDER STATUS CHECK FAILED: \n\n{order_resp} \n\n")
            return 


    def send_orders(self, model_state):
        """
        
    
        Parameters
        ----------
        client : TYPE
            DESCRIPTION.
        model_state : TYPE
            DESCRIPTION.
        df_trade : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        symbol = self.model_config["model_instruments"]["instrument_to_trade"]
        # TODO: intrabar position flipping, this has be list based order system. currently singular order per signal
        # =============================================================================
        # LONGS
        # =============================================================================
        side = None
        # ===================================
        # ENTRY     
        # ===================================
        if model_state["long"]["signal_event"] and model_state["long"]["buy_signal"]:
            position = "long"
            in_position_flag = True
            # CANCEL ALL PENDING ENTRY ORDERS (if any) FIRST
            model_state = self.cancel_order_model_state_update(model_state, position="long", side = "buy")
            model_state = self.cancel_order_model_state_update(model_state, position="short", side = "sell")
            
            # Prepare order params
            side_to_order = "buy"
            price_to_record = model_state[position]["buy_price_expected"]
            price_to_order = price_to_record
            quantity_to_order = model_state[position]["buy_quantity_expected"] 
        # ===================================
        # EXIT     
        # ===================================
        elif model_state["long"]["signal_event"] and model_state["long"]["sell_signal"]:
            position = "long"
            in_position_flag = False
            # CANCEL ALL PENDING EXIT FIRST
            model_state = self.cancel_order_model_state_update(model_state, position="long", side = "sell")
            
            # Prepare order params
            side_to_order= "sell"
            price_to_record = model_state[position]["sell_price_expected"]
            price_to_order = 1/price_to_record
            quantity_to_order = model_state[position]["buy_quantity_filled"] 
            
        # =============================================================================
        # SHORTS       
        # =============================================================================
        # ===================================
        # ENTRY     
        # ===================================
        elif model_state["short"]["signal_event"] and model_state["short"]["sell_signal"]:
            position = "short"
            in_position_flag = True
            # CANCEL ALL PENDING ENTRY ORDERS (if any) FIRST
            model_state = self.cancel_order_model_state_update(model_state, position="short", side = "sell")
            model_state = self.cancel_order_model_state_update(model_state, position="long", side = "buy")
            
            # Prepare order params
            side_to_order= "sell"
            price_to_record = model_state[position]["sell_price_expected"]
            price_to_order = 1/price_to_record
            quantity_to_order = model_state[position]["sell_quantity_expected"] 

        # ===================================
        # EXIT     
        # ===================================
        elif model_state["short"]["signal_event"] and model_state["short"]["buy_signal"]:
            position = "short"
            in_position_flag = False
            # CANCEL ALL PENDING EXIT FIRST
            model_state = self.cancel_order_model_state_update(model_state, position="short", side = "sell")
            
            # Prepare order params
            side_to_order= "buy"
            price_to_record = model_state[position]["buy_price_expected"]
            price_to_order = price_to_record
            quantity_to_order = model_state[position]["sell_quantity_filled"] 
            
        # =============================================================================
        # ELSE- should this be caught?   
        # =============================================================================
        else:
            logger.warning("WHY DID THIS TRIGGER?!")
            pass
        
        
        
        # =========================================================================================================
        # =========================================================================================================
        # SEND ORDER     
        # =========================================================================================================
        # =========================================================================================================
        
        
        # =============================================================================
        # MARKET ORDER
        # =============================================================================
        if self.model_config["run_mode"] == "live" and self.model_config["order_type"] == "market":
            logger.info(f"\n{'!='*50}\n SENDING ORDER for {position} {side}\n \n expected price: {price_to_order}\nquantity: {quantity}\n{'!='*50}\n")
            resp,order_request_timestamp = self.create_market_order(symbol = symbol,
                                                                    side = side_to_order,
                                                                    quantity = quantity)
            self.order_request_timestamp = order_request_timestamp
            order_resp, order_executions_df = self.get_order_status(symbol= symbol, order_request_timestamp = order_request_timestamp)
            # if instant fill, order_resp will be of at most 2 rows
            
            # ===================================
            # Model State update    
            # ===================================
            if (order_resp["execution_status"] == "F").any():
                logger.info("EXECUTION STATUS: FILLED") 
                model_state[position]["signal_event"] = False
                model_state[position]["in_position"] = True
                model_state[position][f"{side}_pending"] = False
                model_state[position]["order_id"] = order_resp["cl_order_id"].iloc[0]
                model_state[position]["buy"] = False
                model_state[position]["sell"] = False
                model_state[position]["actual_price"] = order_resp["average_price"]
                model_state[position]["actual_quantity"] = order_resp["quantity"]
                return model_state
            else:
                logger.info("EXECUTION STATUS: PENDING") 
                model_state[position]["signal_event"] = False
                model_state[position]["in_position"] = False
                model_state[position][f"{side}_pending"] = True
                model_state[position]["order_id"] = order_resp["cl_order_id"].iloc[0]
                model_state[position]["buy"] = False
                model_state[position]["sell"] = False
                model_state[position]["actual_price"] = 0
                model_state[position]["actual_quantity"] = 0
                return model_state
            
            logger.info(f"POST FILL MODEL STATE: \n{pd.DataFrame(model_state)}\n")  
            
            
            
            # ============================================================================= 
            # Update transactions table   
            # ============================================================================= 
            model_state = self.save_to_db(self.model_config, order_resp, model_state)
            
            logger.info(f"POST UPDATE TRANSACTIONS TABLE STATE: \n{pd.DataFrame(model_state)}\n")
        
        # =============================================================================
        # LIMIT ORDER
        # =============================================================================   
        elif self.model_config["run_mode"] == "live" and self.model_config["order_type"] == "limit":
            logger.info(f"\n{'='*30}\nSENDING ORDER for {position} | {side_to_order}\n{'='*30}\n\nexpected price: {price_to_order}\nexpected quantity: {quantity_to_order}\n\n")
            resp,order_request_timestamp = self.create_limit_order(symbol = symbol,
                                                                   side = side_to_order,
                                                                   quantity = quantity_to_order, 
                                                                   price = price_to_order)

            order_resp, order_executions_df = self.get_order_status(symbol= symbol, order_request_timestamp = order_request_timestamp)
            logger.info(f"\n{'='*30}\nOrder Response\n{'='*30}\n\n{order_resp['cl_order_id']}\n")    
            
            # ===================================
            # Model State update    
            # ===================================
            if (order_resp["execution_status"] == "F").any():
                logger.info(f"\n{'='*30}\nEXECUTION STATUS:\n{'='*30}\n\nFILLED\n\n") 
                # if filled, order rep is a dataframe of at least 2 rows
                order_resp = order_resp[order_resp["execution_status"]=="F"].iloc[0]
                self.save_to_db(self.model_config, order_resp, model_state, position=position, side=side_to_order)
                
                # Update model_id for next trade if close position, else dont update
                model_state[position]["model_id"] = model_state[position]["model_id"]+1 if in_position_flag == False else model_state[position]["model_id"]
                model_state[position]["signal_event"] = False
                model_state[position]["in_position"] = in_position_flag
                model_state[position]["buy_signal"] = False
                model_state[position]["sell_signal"] = False
                model_state[position][f"{side_to_order}_pending"] = False # <------------------------------- Pending order set to False since filled immediately
                model_state[position]["order_id"] = order_resp["cl_order_id"]
                model_state[position][f"{side_to_order}_price_filled"] = float(order_resp["average_price"])
                model_state[position][f"{side_to_order}_quantity_filled"] = float(order_resp["quantity"])
                
            else:
                logger.info(f"\n{'='*30}\nEXECUTION STATUS:\n{'='*30}\n\nPENDING\n\n") 
                
                # Dont save to db and update model_state
                model_state[position]["signal_event"] = False
                model_state[position]["buy_signal"] = False
                model_state[position]["sell_signal"] = False
                model_state[position][f"{side_to_order}_pending"] = True  # <------------------------------- Pending order set to True
                model_state[position]["order_id"] = order_resp["cl_order_id"].iloc[-1]
                model_state[position][f"{side_to_order}_price_expected"] = price_to_record
                model_state[position][f"{side_to_order}_quantity_expected"] = quantity_to_order
                
            logger.info(f"\n{'='*30}\nORDER DONE MODEL STATE:\n{'='*30}\n\n{pd.DataFrame(model_state)}\n\n")  
            return model_state
        # =============================================================================
        # PAPER TRADES
        # ============================================================================= 
        else:
            logger.info("PAPER TRADING: ASSUME AUTO-FILLED") 
            order_resp = self.paper_order_resp(quantity, side, model_state)
            
            model_state[position]["in_position"] = True
            model_state[position]["buy"] = False
            model_state[position]["sell"] = False
            model_state[position]["actual_price"] = order_resp["average_price"]
            model_state[position]["actual_quantity"] = order_resp["quantity"]
            
            logger.info(f"POST FILL MODEL STATE: \n{pd.DataFrame(model_state)}\n")  
            
            
            
            # ============================================================================= 
            # Update transactions table   
            # ============================================================================= 
            model_state = self.save_to_db(self.model_config, order_resp, model_state)
            
            logger.info(f"POST UPDATE TRANSACTIONS TABLE STATE: \n{pd.DataFrame(model_state)}\n")
            return model_state
    
    def paper_order_resp(self, quantity, side, price):
        
    
        order_resp = copy.deepcopy(order_resp_default)
        order_timestamp = time.time_ns() // 1000000
        order_datetime = dt.fromtimestamp(order_timestamp/1000)
        order_resp["order_id"] = f"B{order_timestamp}DTF_Paper"
        order_resp["cl_order_id"] = order_timestamp
        order_resp["quantity"] = quantity
        order_resp["side"] = 1 if side == "buy" else 2
        order_resp["symbol"] = self.model_config["model_instruments"]["instrument_to_trade"]
        order_resp["execution_status"] = "F"
        order_resp["order_status"] = 2
        order_resp["last_price"] = price
        order_resp["average_price"] = price
        order_resp["execution_message"] = order_resp["average_price"]
        order_resp["transaction_time"] = order_datetime
        order_resp["created_at"] = order_datetime
        order_resp["updated_at"] = order_datetime
    
        return order_resp
    

    
# ==========================================================================================================================================================
# States and transaction table methods
# ==========================================================================================================================================================
    
    
    def load_positions_from_transactions_table(self, model_config=None) -> dict:
        """
        

        Parameters
        ----------
        model_config : TYPE, optional
            DESCRIPTION. If none then loads entire transactions table, else then selects by model_name in model_config

        Returns
        -------
        dict
            DESCRIPTION. transactions table as dataframe

        """
        transactions_df = db_manager.read_table(table_name="transactions")
        
        if model_config is None:
            return transactions_df
        else:
            return transactions_df[transactions_df["model_name"]==model_config["model_name"]]
    
    
    
    def load_model_state(self, model_config=None, transactions_df=None, model_state=None) -> dict:
        """
        Model state is wholly dependant on what is recorded in transactions table
        - this is predicated on successful recording after order fills have been made
        - if model_state is not provided, initialise model state from transactions table
        - if model_state is provided, this is used for updating model_id 
        
        Description:
        Checks model_id if there are odd numbers of model_id in tail end. if so then pick up position, and update model_id
            
        
        """
        if not model_state:
            model_state = copy.deepcopy(model_state_default)
        
        if transactions_df is None:
            transactions_df = self.load_positions_from_transactions_table()
            if len(transactions_df) == 0:
                return model_state
            if model_config is not None:
                account = model_config["account"]
                model_name = model_config["model_name"]
                # select only those from model config
                transactions_df = transactions_df[transactions_df["model_name"]==model_name]
                transactions_df = transactions_df[transactions_df["account"]==account]
            
            
        if len(transactions_df) == 0:
            return model_state
        
        
        for position in ["long", "short"]:
            t_df_i = transactions_df[transactions_df["position"]==position]
            if len(t_df_i) == 0:
                pass
                
            elif len(t_df_i) == 1:
                last_id = t_df_i["model_trade_id"].iloc[-1]
                last_side =  t_df_i["side"].iloc[-1]
                
                model_state[position]["model_id"] = last_id
                model_state[position]["in_position"] = True
                model_state[position][f"{last_side}_quantity_filled"] = t_df_i["quantity"].iloc[-1]
                model_state[position][f"{last_side}_price_filled"] = t_df_i["average_price"].iloc[-1]
                model_state[position][f"{last_side}_price_expected"] = t_df_i["expected_price"].iloc[-1]
            else:
                """
                If there are more than 1 trades done,
                check all trades of that position and infer open position (if any) to ascertain model_state
                """
                last_id = t_df_i["model_trade_id"].iloc[-1]
                last_side =  t_df_i["side"].iloc[-1]
                
                # this subset df should start with buy and end with sell, now check if sells quantity - buy quantity = 0 if not then this is an open position
                last_t_df_i = t_df_i[t_df_i["model_trade_id"]==last_id]
                if position == "long":
                    buy_sell_diff = last_t_df_i[last_t_df_i["side"]=="buy"]["quantity"].iloc[0] - last_t_df_i[last_t_df_i["side"]=="sell"]["quantity"].sum() 
                elif position == "short":
                    buy_sell_diff = last_t_df_i[last_t_df_i["side"]=="sell"]["quantity"].iloc[0] - last_t_df_i[last_t_df_i["side"]=="buy"]["quantity"].sum() 
                logger.info(f"Inferred open_position buy sell diff: {buy_sell_diff}")

                if buy_sell_diff > 0:
                    model_state[position]["model_id"] = last_id
                    model_state[position]["in_position"] = True
                    model_state[position][f"{last_side}_quantity_filled"] = buy_sell_diff
                    model_state[position][f"{last_side}_price_filled"] = t_df_i["average_price"].iloc[-1]
                    model_state[position][f"{last_side}_price_expected"] = t_df_i["expected_price"].iloc[-1]
                elif buy_sell_diff == 0: 
                    # NO OPEN POSITION! Update only model_id 
                    model_state[position]["model_id"] = last_id+1
                else:
                    logger.error("buy_sell_diff is negative! sold more than bought?? check table edit methods")
        
        return model_state
        
    
        
        
    def update_model_state(self, model_state, model_config):
        """
        Checks executions table for order fills and ammend model_state if so
        THIS REQUIRES order_id (cl_order_id) to be previously set in first order 
        """
        symbol= self.model_config["model_instruments"]["instrument_to_trade"]
        for position in ["long","short"]:
            if model_state[position]["order_id"] == 0:
                continue
            order_executions_df_by_id = self.get_order_status_using_cl_order_id(symbol=symbol, cl_order_id=model_state[position]["order_id"])
            filled_order_executions_df_by_id = order_executions_df_by_id[order_executions_df_by_id["execution_status"]=="F"]
            if len(filled_order_executions_df_by_id) == 0:
                # NOT FILLED YET so wait
                continue
            else:
                order_resp = filled_order_executions_df_by_id.iloc[-1,:]
                
            for side in ["buy","sell"]:
                if model_state[position][f"{side}_pending"] and (order_executions_df_by_id["execution_status"] == "F").any():
                    # ORDER PENDING WAS FILLED!! 
                    model_state[position][f"{side}_pending"] = False
                    model_state[position]["order_id"] = 0
                    
                    # update db
                    self.save_to_db(model_config, order_resp, model_state, position=position, side=side)
                    # ORDER FILL on long buy or short sell --> position has been OPENED
                    if (side == "buy" and position=="long") or (side == "sell" and position=="short"):
                        model_state[position]["in_position"] = True
                    # ORDER FILL on long sell or short buy --> position has been CLOSED
                    elif (side == "sell" and position=="long") or (side == "buy" and position=="short"):
                        model_state[position]["in_position"] = False
                        model_state[position]["model_id"]+=1
                    
                    model_state[position][f"{side}_price_filled"] = float(filled_order_executions_df_by_id["average_price"].iloc[0])
                    model_state[position][f"{side}_quantity_filled"] = float(filled_order_executions_df_by_id["quantity"].iloc[0])
                    
                    # Update transactions table on filled order
                    
        
        # STILL PENDING... wait some more
        return model_state 
    
    
    
    
    def save_to_db(self,model_config, order_resp, model_state, position, side):
        """
        Records FILLED orders to db - only called when order fill is confirmed! 
        requires
        - model_config for account and model details
        - order_resp for order details
        - model_state for position details
        """
            
        metric_dict_to_insert = {"order_id":model_state[position]["model_id"],
                                 "cl_order_id": model_state[position]["order_id"],
                                 "execution_id": order_resp["execution_id"],
                                 "account": model_config["account"],
                                 "model_name":model_config["model_name"],
                                 "model_trade_id": float(model_state[position]["model_id"]),
                                 "symbol": model_config["model_instruments"]["instrument_to_trade"],
                                 "reporting_currency": model_config["reporting_currency"],
                                 "instrument_type": model_config["instrument_type"],
                                 "order_type": model_config["order_type"],
                                 "position": position,
                                 "side": side,
                                 "quantity": float(order_resp["quantity"]),
                                 "average_price": float(order_resp[f"average_price"]),
                                 "expected_price": float(model_state[position][f"{side}_price_expected"]),
                                 "created_at": order_resp["created_at"],
                                 "updated_at": order_resp["updated_at"],
                                 }     
            
        metrics_to_insert = [list(metric_dict_to_insert.values())]
        
        logger.info("SAVING FILLED ORDER TO DB")
        db_manager.write_to_table(metrics_to_insert)
        




    
    def consolidate_pnl(self,model_config, transactions_df = None):
        model_state = self.load_model_state(model_config=model_config)
        print(f"{'=='*10}\nINITIAL MODEL STATE:\n{'=='*10} \n{pd.DataFrame(model_state)}\n")
        
        if transactions_df is None:
            transactions_df = self.load_positions_from_transactions_table(model_config = model_config)
        
        # This method will reconstruct trade_df as in backtest and then merge to df_sig to produce pnl series
        trades_df_list = []
        t1=time.time()
        for position in ["long", "short"]:
            
            L_orders0 = transactions_df[transactions_df["position"]==position].copy()
            
            L_orders = pd.DataFrame()
            
            L_orders["created_at"]=L_orders0["created_at"]
            L_orders["open_at"] = np.nan
            L_orders["close_at"] = np.nan
            L_orders["model_name"] = L_orders0["model_name"]
            L_orders["account"] = L_orders0["account"]
            L_orders["symbol"] = L_orders0["symbol"]
            L_orders["instrument_type"] = L_orders0["instrument_type"]
            L_orders["reporting_currency"] = L_orders0["reporting_currency"]
            
            L_orders["position"] = position
            L_orders["model_trade_id"] = L_orders0["model_trade_id"]
            L_orders["positions"] = np.where(L_orders0["side"]=="buy", 1, np.nan)
            L_orders["actual_entry_price"] = np.where(L_orders0["side"]=="buy", L_orders0["average_price"], np.nan)
            L_orders["expected_entry_price"] = np.where(L_orders0["side"]=="buy", L_orders0["expected_price"], np.nan)
            L_orders["cost"] = L_orders0["quantity"]
            
            L_orders["actual_qty"] = L_orders["cost"]/ L_orders["actual_entry_price"]
            L_orders["actual_qty"].fillna(method="ffill",inplace=True)
            L_orders["expected_qty"] = L_orders["cost"]/ L_orders["expected_entry_price"]
            L_orders["expected_qty"].fillna(method="ffill",inplace=True)
            
            L_orders["actual_exit_price"] = np.where(L_orders0["side"]=="sell", L_orders0["average_price"], np.nan)
            L_orders["expected_exit_price"] = np.where(L_orders0["side"]=="sell", L_orders0["expected_price"], np.nan)
            
            L_orders["actual_fees"] = 0
            L_orders["expected_fees"] = model_config["equity_settings"]["fee"]
            
            L_orders["actual_pnl_pct"] = 0
            L_orders["expected_pnl_pct"] = 0
            L_orders["actual_rpnl"] = 0
            L_orders["expected_rpnl"] = 0
            # L_orders["trail"] = 0
            # L_orders["comment"] = 0 
            
            # Loop through model_trade id and perform consolidations
            for L_id in np.unique(L_orders["model_trade_id"]):
                L_orders_i = L_orders[L_orders["model_trade_id"]==L_id]
                indexes = L_orders_i.index
                actual_entry_price = L_orders_i["actual_entry_price"].dropna().iloc[0]
                expected_entry_price = L_orders_i["expected_entry_price"].dropna().iloc[0]
                try:
                    actual_exit_price = L_orders_i["actual_exit_price"].dropna().iloc[0]
                except Exception as e:
                    actual_exit_price = actual_entry_price
                
                try:
                    expected_exit_price = L_orders_i["expected_exit_price"].dropna().iloc[0]
                except Exception as e:
                    expected_exit_price = expected_entry_price
                    
                actual_qty = L_orders_i["actual_qty"].dropna().iloc[0]
                expected_qty = L_orders_i["expected_qty"].dropna().iloc[0]

                if position == "long":
                    L_orders.at[indexes[-1],"actual_pnl_pct"] = actual_exit_price/actual_entry_price - 1
                    L_orders.at[indexes[-1],"actual_rpnl"] = (actual_exit_price - actual_entry_price) * actual_qty
                    
                    L_orders.at[indexes[-1],"expected_pnl_pct"] = expected_exit_price/expected_entry_price - 1
                    L_orders.at[indexes[-1],"expected_rpnl"] = (expected_exit_price - expected_entry_price) * expected_qty
                    
                    
                elif position == "short":
                    L_orders.at[indexes[-1],"actual_pnl_pct"] = actual_entry_price/actual_exit_price - 1 
                    L_orders.at[indexes[-1],"actual_rpnl"] = (actual_entry_price - actual_exit_price) * actual_qty
                    
                    L_orders.at[indexes[-1],"expected_pnl_pct"] = expected_entry_price/expected_exit_price - 1
                    L_orders.at[indexes[-1],"expected_rpnl"] = (expected_entry_price - expected_exit_price ) * expected_qty
                    
                L_orders.at[indexes[-1],"open_at"] = L_orders.loc[indexes[0],"created_at"]
                L_orders.at[indexes[-1],"close_at"] = L_orders.loc[indexes[-1],"created_at"]
            
        
            L_orders[['positions', 'actual_entry_price', 'expected_entry_price']] = L_orders[['positions', 'actual_entry_price', 'expected_entry_price']].fillna(method="ffill")
            L_orders.dropna(inplace=True)
            L_orders.drop(columns=["created_at"], inplace=True)
            L_orders.sort_values(by="open_at", inplace=True)
            
            trades_df_list.append(L_orders)
            
        trades_df = pd.concat(trades_df_list,axis=0)
        trades_df.sort_values(by="open_at", inplace=True)
        t2 = round(time.time() - t1,2)
        print(f"time taken: {t2}s")
        
        return trades_df
        
#%% Order flow test 
# 1) place limit order
# 2) wait till bar done (1m)
# 3) if order filled ---> update model_state else, wait. 
# 4) if order not filled ---> wait till fill OR new signal appears then cancel previous order
if __name__ == "__main__":
    #%% DEBUGGING BOT

    #%%
    from config.load_config import load_config_file
    
    # Load model name from config-dev
    MODEL_NAME = load_config_file('./config/config-dev.json')['model_name']
    # Load model config from given model name in config-dev.json
    model_config = load_config_file(f'./config/{MODEL_NAME}.json')
    
    
    client = KGI_Client(model_config)
    
    order_executions_df = client.get_order_executions()
    check = order_executions_df[order_executions_df["transaction_time"]==order_executions_df["transaction_time"].iloc[-1]]
    check_t_df = client.load_positions_from_transactions_table(model_config)
    #%% load model states
    model_state = client.load_model_state(model_config, model_state=None)
    print(pd.DataFrame(model_state))
    #%% delete 
    # from data_manager import db_manager
    # db_manager.delete_rows(table_name = "transactions", where = "model_name='EURUSD_1m_test'")
    
    #%% MARKET ORDER  =================================================
    symbol="USDSGD"
    side = "buy"
    quantity = 100
    resp,order_request_timestamp = client.create_market_order(symbol=symbol,
                                            side=side, 
                                            quantity=quantity)
    
    # =============================================================================
    #%% LIMIT ORDER
    # =============================================================================
        
    symbol="USDSGD"
    side = "buy"
    price = 0.5
    quantity = 1000
    model_state = client.load_model_state(model_config)

    resp,order_request_timestamp = client.create_limit_order(symbol=symbol,
                                            side=side, 
                                            price= price,
                                            quantity=quantity)
    
    
    # check order fill
    # retry this every minute until order is filled OR until new signal comes then cancel last order
    order_resp, order_executions_df = client.get_order_status(symbol=symbol, order_request_timestamp=order_request_timestamp)
    order_id = order_resp["cl_order_id"]
    
    #%% cancel order check
    side = "sell"
    order_id_to_cancel = str(1675395724106)
    quantity = 12796.32
    cancel_order_resp,order_cancel_t = client.cancel_order(symbol="EURUSD", side = side,quantity=quantity, order_id = order_id_to_cancel)
    
    
    #%%
    import requests
    import json
    
    url = "http://18.140.28.101:8081/api/"
    auth_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiUGV0ZXIgV3UgV2VpIiwiaWF0IjoxNjY5NzA3ODcxfQ.iPP0jkN9xVjqFSNHZxhkE6zeKhnuPOwxv61LvxhKRi8'
    headers = {'Authorization': 'Bearer '+ auth_token}
    
    
    if side == "buy":
        order_dict = {"from":symbol[:3].upper(), 
                      "to":symbol[-3:].upper(),
                      "action": side.lower(),
                      "qty": quantity,
                      "order_id": order_id_to_cancel}
    elif side == "sell":
        order_dict = {"from":symbol[-3:].upper(), 
                      "to":symbol[:3].upper(),
                      "action": side.lower(),
                      "qty": quantity,
                      "order_id": order_id_to_cancel}

    # try:
    resp = requests.post(url=url + "order/cancel", json=order_dict, headers=headers)
    print(f"{resp}")
    
    
    
    #%% check order fill
    # retry this every minute until order is filled OR until new signal comes then cancel last order
    order_resp, order_executions_df = client.get_order_status(symbol="EURUSD", order_request_timestamp=order_request_timestamp)
    order_id = order_resp["cl_order_id"]
    
    #%% Subsequent checks on order status
    
    symbol="EURUSD"
    side = "buy"
    price = 1.08900
    quantity = 10000
    model_state = client.load_model_state(model_config)

    resp2,order_request_timestamp2 = client.create_limit_order(symbol=symbol,
                                            side=side, 
                                            price= price,
                                            quantity=quantity)
    
    #%%
    order_resp2, order_executions_df2 = client.get_order_status(symbol=symbol, order_request_timestamp=order_request_timestamp2)
    order_id2 = order_resp2["cl_order_id"]
    #%% Subsequent checks on order status
    

    #%% CANCEL ORDER test

    cancel_order_resp = client.cancel_order(symbol=symbol, side = side,quantity=quantity, order_id = order_id)
    #%%
    cl_order_id = 1674725466561
    order_executions_df = client.get_order_status_using_cl_order_id(symbol="EURUSD", cl_order_id=cl_order_id)
         
         
         
         
    #%% market order test
    
    symbol="EURUSD"
    side = "buy"
    quantity = 0.4 #143633.6
    resp,order_request_timestamp = client.create_market_order(symbol=symbol,
                                            side=side, 
                                            quantity=quantity)     
         
    
