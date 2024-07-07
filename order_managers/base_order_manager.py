

from abc import ABC, abstractmethod



class base_order_manager(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
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

    @abstractmethod
    def create_market_order(self,
                   model_state,
                   position, 
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


        model_state[position]["in_position"] = True
        model_state[position][f"{side}_now"] = True 
        model_state[position][f"{side}_pending"] = False
        model_state[position][f"{side}_price_filled"] = model_state["close"]
        model_state[position][f"{side}_time"] = model_state[position]["time"]


        return model_state
    


    @abstractmethod
    def create_limit_order(self,
                   model_state,
                   position, 
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

        # update model_state
        model_state[position]["in_position"] = False
        model_state[position][f"{side}_now"] = True 
        model_state[position][f"{side}_pending"] = True

        return model_state

    @abstractmethod
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
        # update model_state
        for pos in ["L","S"]:
            if model_state[pos]["entry_now"]:
                model_state = self.create_market_order(model_state, pos, "entry")
            elif model_state[pos]["exit_now"]:
                model_state = self.create_market_order(model_state, pos, "exit")
            elif model_state[pos]["entry_pending"]:
                model_state = self.create_limit_order(model_state, pos, "entry")
            elif model_state[pos]["exit_pending"]:
                model_state = self.create_limit_order(model_state, pos, "exit")
            else:
                continue

        return model_state

    @abstractmethod
    def update_model_state(self,
                           model_state,
                           order_type = "market"):
        
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
        """
        if order_type == "limit":
            model_state = self.check_for_limit_order_fills(model_state)
        elif order_type == "market":
            model_state = self.check_for_market_order_fills(model_state)

        # Check cancel orders if necessary
        # Check 
        return model_state

    @abstractmethod
    def check_for_market_order_fills(self, model_state):
        for pos in ["L","S"]:
            if model_state[pos]["entry_now"]:
                if pos == "L":
                    model_state[pos]["in_position"] = True
                    model_state[pos]["positions"] = 1
                    model_state[pos]["entry_pending"] = False
                    model_state[pos]["entry_now"] = False


                elif pos == "S":
                    model_state[pos]["in_position"] = True
                    model_state[pos]["positions"] = -1
                    model_state[pos]["entry_pending"] = False
                    model_state[pos]["entry_now"] = False

            if model_state[pos]["exit_now"]:
                if pos == "L":
                    # UPDATE MODEL STATE
                    model_state[pos]["in_position"] = False
                    model_state[pos]["positions"] = 0
                    model_state[pos]["exit_pending"] = False
                    model_state[pos]["exit_now"] = False

                elif pos == "S":
                    # UPDATE MODEL STATE
                    model_state[pos]["in_position"] = False
                    model_state[pos]["positions"] = 0
                    model_state[pos]["exit_pending"] = False
                    model_state[pos]["exit_now"] = False
        return model_state

    @abstractmethod
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