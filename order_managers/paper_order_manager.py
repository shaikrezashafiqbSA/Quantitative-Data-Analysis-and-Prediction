
import copy
import time
import datetime

from utils.get_time import get_dt_times_now
from order_managers.order_resp_default import order_resp_default
from logger.logger import setup_logger
from db_managers import KGI_db_manager as db_manager
from datetime import datetime as dt
from model_states.model_state_default import model_state_default
logger = setup_logger(logger_file_name="trading_bot")


class Paper_Order_Manager():
    def __init__(self, model_config, save_to_db_flag = True):
        self.model_config = model_config
        self.save_to_db_flag = save_to_db_flag

    def cancel_order(self, 
                     model_state,
                     position, 
                     side):
        # SEND ORDER HERE ASYNC
        """
        SEND CANCEL ORDER 
        order_dict = {"from":symbol[-3:].upper(), 
                "to":symbol[:3].upper(),
                "action": side.lower(),
                "qty": quantity,
                "order_id": str(order_id)}

        await order_manager.cancel_order(order_dict)

        """
        # update model_state
        model_state[position]["in_position"] = False
        model_state[position][f"{side}_now"] = False
        model_state[position][f"{side}_pending"] = False

        return model_state

    def create_market_order(self,
                   model_state,
                   pos, 
                   side):
        """
        # Order sending here

        symbol = model_state[position]["symbol"]
        side_to_order = model_state[position][f"{side}"]
        quantity_to_order = model_state[position][f"{side}_quantity"]
        price_to_order = model_state[position][f"{side}_price"]
        
        resp,order_request_timestamp = self.create_limit_order(symbol = symbol,
                                                        side = side_to_order,
                                                        quantity = quantity_to_order, 
                                                        price = price_to_order)
        """
        logger.info("PAPER TRADING: ASSUME AUTO-FILLED") 
        t1 = time.time_ns()
        if side == "entry":
            quantity = model_state[pos][f"{side}_quantity_expected"]
            price_to_order =  model_state[pos][f"{side}_price_expected"]
        elif side == "exit":
            quantity = model_state[pos][f"exit_quantity_expected"]
            price_to_order =  model_state[pos][f"{side}_price_expected"]
        else:
            raise Exception("side must be either 'entry' or 'exit'")
        
        resp = self.paper_order_resp(quantity, side, price_to_order)


        return resp, t1
    

    # def create_limit_order(self,
    #                model_state,
    #                position, 
    #                side):
    #     """
    #     # Order sending here

    #     symbol = model_state[position]["symbol"]
    #     side_to_order = model_state[position][f"{side}"]
    #     quantity_to_order = model_state[position][f"{side}_quantity"]
    #     price_to_order = model_state[position][f"{side}_price"]
        
    #     resp,order_request_timestamp = self.create_limit_order(symbol = symbol,
    #                                                     side = side_to_order,
    #                                                     quantity = quantity_to_order, 
    #                                                     price = price_to_order)
    #     """

    #     # update model_state
    #     model_state[position]["in_position"] = False
    #     model_state[position][f"{side}_now"] = True 
    #     model_state[position][f"{side}_pending"] = True

    #     return model_state



    def send_orders(self, model_state):
        # SEND ORDER HERE ASYNC
        """
        SEND CANCEL ORDER 
        order_dict = {"from":symbol[-3:].upper(), 
                "to":symbol[:3].upper(),
                "action": side.lower(),
                "qty": quantity,
                "order_id": str(order_id)}

        await order_manager.cancel_order(order_dict)

        """
        # if self.simulate_bid_ask_taking:
        #     # get bid ask prices from rest api
        #     bid_ask = self.get_bid_ask(symbol)
        # put into model_state bid ask
        #     pass
            
        # Main body send_orders
           
        # old main body send_orders update model_state
        for pos in ["L","S"]:
            if model_state[pos]["entry_now"]:
                side_to_order = "buy" if pos == "L" else "sell"
                side = "entry" if pos == "L" else "exit"
                resp, t1 = self.create_market_order(model_state, pos, "entry")
                logger.info(f"!!!!! {pos} ENTRY ORDER SENT !!!!!")
            elif model_state[pos]["exit_now"]:
                side_to_order = "sell" if pos == "L" else "buy"
                side = "entry" if pos == "L" else "exit"
                resp, t1 = self.create_market_order(model_state, pos, "exit")
                logger.info(f"!!!!! {pos} EXIT ORDER SENT !!!!!")

                model_state = self.post_process_orders(model_state, pos, side)
            # elif model_state[pos]["entry_pending"]:
            #     side_to_order = "buy" if pos == "L" else "sell"
            #     model_state = self.create_limit_order(model_state, pos, "entry")
            #     logger.info(f"!!!!! {pos} entry ORDER SENT !!!!!")
            # elif model_state[pos]["exit_pending"]:
            #     side_to_order = "sell" if pos == "L" else "buy"
            #     model_state = self.create_limit_order(model_state, pos, "exit")
            #     logger.info(f"!!!!! {pos} exit ORDER SENT !!!!!")
            else:
                continue
            logger.info(f"ORDER RESPONSE: {resp} --->\n {model_state}")
            if self.save_to_db_flag:
                self.save_to_db(self.model_config, 
                                resp, 
                                model_state, 
                                position=pos, 
                                side=side_to_order)
        return model_state
    
    def post_process_orders(self, model_state, position, side):
        # POST PROCESS ORDERS HERE
        model_state[position]["in_position"] = True
        model_state[position][f"{side}_now"] = True 
        model_state[position][f"{side}_pending"] = False
        model_state[position][f"{side}_price"] = model_state["close"]
        model_state[position][f"{side}_time"] = model_state[position]["time"]
        return model_state
    
    def paper_order_resp(self, quantity, side, price):
        
    
        order_resp = copy.deepcopy(order_resp_default)
        order_timestamp = time.time_ns() // 1000000
        order_datetime = dt.fromtimestamp(order_timestamp/1000)
        order_resp["order_id"] = f"B{order_timestamp}DTF_Paper"
        order_resp["cl_order_id"] = order_timestamp
        order_resp["quantity"] = quantity
        order_resp["side"] = 1 if side == "buy" else 2
        order_resp["symbol"] = self.model_config["instrument_to_trade"]
        order_resp["execution_status"] = "F"
        order_resp["order_status"] = 2
        order_resp["last_price"] = price
        order_resp["average_price"] = price
        order_resp["execution_message"] = order_resp["average_price"]
        order_resp["created_at"] = order_datetime
        order_resp["updated_at"] = order_datetime
    
        return order_resp
    


    def update_model_state(self,
                           model_state,
                           model_config,
                           order_type = "market", replay_dt=None):
        
        """
        if model_state[position]["order_id"] == 0:
            continue
        order_executions_df_by_id = self.get_order_status_using_cl_order_id(symbol=symbol, cl_order_id=model_state[position]["order_id"])
        filled_order_executions_df_by_id = order_executions_df_by_id[order_executions_df_by_id["execution_status"]=="F"]
        if len(filled_order_executions_df_by_id) == 0:
            # NOT FILLED YET so wait
            continue
        else:
            order_resp = filled_order_executions_df_by_id.iloc[-1,:]
        
        # update model_state
        FIRSTLY. update model_state["time"] to time of this function call
        """
        # 1. update model_state["time"] to time of this function call
        if replay_dt is None:
            dt_now_UTC,_ = get_dt_times_now()
            t_to_insert = dt_now_UTC
        else: 
            t_to_insert = str(replay_dt)
        for pos in ["L","S"]:
            model_state[pos]["time"] = t_to_insert

        #  ensure order_type is either "market" or "limit"
        assert order_type in ["market", "limit"], "order_type must be either 'market' or 'limit'"
        if order_type == "limit":
            model_state = self.check_for_limit_order_fills(model_state)
        elif order_type == "market":
            model_state = self.check_for_market_order_fills(model_state)

        # Check cancel orders if necessary
        # Check 
        return model_state


    def check_for_market_order_fills(self, model_state):
        for pos in ["L", "S"]:
            if model_state[pos]["entry_now"]:
                model_state[pos]["in_position"] = True
                model_state[pos]["positions"] = 1 if pos == "L" else -1
                model_state[pos]["entry_pending"] = False
                model_state[pos]["entry_now"] = False

            if model_state[pos]["exit_now"]:
                model_state[pos]["in_position"] = False
                model_state[pos]["positions"] = 0
                model_state[pos]["exit_pending"] = False
                model_state[pos]["exit_now"] = False

        return model_state


    def check_for_limit_order_fills(self, model_state):
        for pos in ["L","S"]:
            if model_state[pos]["entry_pending"]:
                if pos == "L":
                    try:
                        if model_state["low"] < model_state[pos]["entry_price"] :
                            model_state[pos]["entry_price_filled"] = model_state[pos]["entry_price"]
                        elif model_state["high"] > model_state[pos]["entry_price_max"]:
                            model_state[pos]["entry_price_filled"] = model_state[pos]["entry_price_max"]
                        else:
                            model_state[pos]["entry_price_filled"] = (model_state[pos]["entry_price"]+model_state[pos]["entry_price_max"]) / 2
                    except Exception as e:
                        print(f"An error occured while trying to fill entry price for {pos} position: {e}")
                    finally:
                        # UPDATE MODEL STATE
                        model_state[pos]["in_position"] = True
                        model_state[pos]["positions"] = 1
                        model_state[pos]["entry_pending"] = False


                elif pos == "S":
                    try:
                        if model_state["high"] > model_state[pos]["entry_price"] :
                            model_state[pos]["entry_price_filled"] = model_state[pos]["entry_price"]
                        elif model_state["low"] < model_state[pos]["entry_price_min"]:
                            model_state[pos]["entry_price_filled"] = model_state[pos]["entry_price_min"]
                        else:
                            model_state[pos]["entry_price_filled"] = (model_state[pos]["entry_price"]+model_state[pos]["entry_price_min"]) / 2
                    except Exception as e:
                        print(f"An error occured while trying to fill entry price for {pos} position: {e}")
                    finally:
                        # UPDATE MODEL STATE
                        model_state[pos]["in_position"] = True
                        model_state[pos]["positions"] = -1
                        model_state[pos]["entry_pending"] = False

            if model_state[pos]["exit_pending"]:
                if pos == "L":
                    try:
                        if model_state["high"] > model_state[pos]["exit_price"] :
                            model_state[pos]["exit_price_filled"] = model_state[pos]["exit_price"]
                        elif model_state["low"] < model_state[pos]["exit_price_min"]:
                            model_state[pos]["exit_price_filled"] = model_state[pos]["exit_price_min"]
                        else:
                            model_state[pos]["exit_price_filled"] = (model_state[pos]["exit_price"]+model_state[pos]["exit_price_min"]) / 2
                    except Exception as e:
                        print(f"An error occured while trying to fill exit price for {pos} position: {e}")
                    finally:
                        # UPDATE MODEL STATE
                        model_state[pos]["in_position"] = False
                        model_state[pos]["positions"] = 0
                        model_state[pos]["exit_pending"] = False

                elif pos == "S":
                    try:
                        if model_state["low"] < model_state[pos]["exit_price"] :
                            model_state[pos]["exit_price_filled"] = model_state[pos]["exit_price"]
                        elif model_state["high"] > model_state[pos]["exit_price_max"]:
                            model_state[pos]["exit_price_filled"] = model_state[pos]["exit_price_max"]
                        else:
                            model_state[pos]["exit_price_filled"] = (model_state[pos]["exit_price"]+model_state[pos]["exit_price_max"]) / 2
                    except Exception as e:
                        print(f"An error occured while trying to fill exit price for {pos} position: {e}")
                    finally:
                        # UPDATE MODEL STATE
                        model_state[pos]["in_position"] = False
                        model_state[pos]["positions"] = 0
                        model_state[pos]["exit_pending"] = False

        return model_state
    

    def load_model_state(self, model_config=None, transactions_df=None, model_state = None) -> dict:
        if model_state is None:
            logger.info(f"No model state saved ---> Creating new model state ...")
            model_state = {"running_signal":None,
                                "open":None,
                                "high":None,
                                "low":None,
                                "close":None,
                                "bid":None,
                                "ask":None,
                                "model" :None,
                                "pair" : None,
                                "time" : None,
                                "time SGT": None,
                                "tradable_times": None,
                                "L": {"model": None,
                                        "pair": None,
                                        "time": None,
                                        "time SGT": None,   
                                        "tradable_times":None,
                                        "in_position":False,
                                        "positions":0,
                                        "side":None,
                                        "model_id":None,
                                        "order_id":None,
                                        "signal_event": False,

                                        "entry_now": False,
                                        "entry_pending":False,
                                        "entry_price": None,
                                        "entry_price_max": None,
                                        "entry_price_expected": None,
                                        "entry_quantity":None,
                                        "entry_quantity_expected":None,
                                        "entry_time": None,

                                        "exit_now":False, 
                                        "exit_pending":False,
                                        "exit_price":None,
                                        "exit_price_min": None,
                                        "exit_price_expected": None,
                                        "exit_quantity":None,
                                        "exit_quantity_expected":None,
                                        "exit_time": None,
                                        },
                                "S": {"model": None,
                                        "pair": None,
                                        "time": None,
                                        "time SGT": None,   
                                        "tradable_times":None,
                                        "in_position":False,
                                        "positions":0,
                                        "side":None,
                                        "model_id":None,
                                        "order_id":None,
                                        "signal_event": False,

                                        "entry_now":False,
                                        "entry_pending":False,
                                        "entry_price": None,
                                        "entry_price_min": None,
                                        "entry_price_expected": None,
                                        "entry_quantity":None,
                                        "entry_quantity_expected":None,
                                        "entry_time": None,

                                        "exit_now":False, 
                                        "exit_pending":False,
                                        "exit_price":None,
                                        "exit_price_max": None,
                                        "exit_price_expected": None,
                                        "exit_quantity":None,
                                        "exit_quantity_expected":None,
                                        "exit_time": None,
                                        }, 
                                }
            
            return model_state
        
    def consolidate_pnl(self, model_config, transactions_df = None):
        # TODO: This suffice for paper trades but WIP: rest api bid-ask and simulate trade to fill transactions_df with "filled_price" and "filled_quantity" trades
        
        return transactions_df
    



    def save_to_db(self, 
                   model_config,
                    order_resp, 
                    model_state, 
                    position, 
                    side):
        """
        Records FILLED orders to db - only called when order fill is confirmed! 
        requires
        - model_config for account and model details
        - order_resp for order details
        - model_state for position details
        """
        if side == "buy":
            side_model = "entry"
        elif side == "sell":
            side_model = "exit"
        logger.info(f"\n\nSave to DB: model_state {model_state}\n\n")  


        if position == "L":
            quantity_size = model_config["long_notional"]
        elif position == "S":   
            quantity_size = model_config["short_notional"]    

        try:
            expected_price = float(model_state[position][f"{side_model}_price_expected"])
        except Exception as e:
            expected_price = 0
        # populate metrics_dict_to_insert with the following dict, if error fill with 0
        default_account = "default_account"
        default_model = "default_model"

        # get todays
        dt_now =  datetime.datetime.now()

        fallback_metrics_dict_to_insert = {"model_name": "default_model",
                                            "trade_currency": "XXX",
                                            "trade_type": "XXX",
                                            "position": "XX",
                                            "side": "XXXX",
                                            "quantity": 0,
                                            "pair": "XXX",
                                            "price" : 0,
                                            }
        
        # update fall_back_metrics_dict_to_insert
        fallback_metrics_dict_to_insert.update({
                                "model_name":model_config["model_name_to_DB"],
                                "trade_currency": model_config["reporting_currency"],
                                "trade_type": model_config["order_type"],
                                "position": position, # "L" or "S"
                                "side": side,        # 'buy' or 'sell'
                                "quantity": quantity_size,
                                "pair": model_config["instrument_to_trade"],
                                "price": order_resp["average_price"],
                                })

        
            
        metrics_to_insert = [list(fallback_metrics_dict_to_insert.values())]

        
        logger.info("SAVING SIGNALS TO DB")
        db_manager.write_to_table(metrics_to_insert, table = "signals")



    