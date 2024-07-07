
def build_params(params_payload_to_add : dict = None):
        data_params_payload = {}
        # =======================================================================================================
        # Core params
        # =======================================================================================================

        non_continuous_vars = ['instrument_to_trade','backtest_window','signal_to_trade',
                                'signal_function','model_name','tradable_times','days_of_the_week_to_trade',
                                'timeframe_to_trade','show_plots', 'show_plots_TP','show_rolling_SR',
                                'run_signals_w_TP', 'mutate_signals_w_TP','position_sizing_to_trade',
                                'fee', 'slippage','long_equity','long_notional','short_equity','short_notional',
                                'tide_dynamic_run','tide_dynamic_alt','z_dynamic_run','z_dynamic_alt','sig_lag',
                                'reduce_only','volume_to_trade','kline_to_trade',
                                'MFI_sharpe_windows','MFI_strong_windows','MFI_weak_windows','MFI_flat_windows','tp_position_dict'
                                ]
        # Non dependent variables (for utility only)
        data_params_payload['show_plots'] = [False,]
        data_params_payload['show_plots_TP'] = [False,]
        data_params_payload['show_rolling_SR'] = [False,]
        data_params_payload['mutate_signals_w_TP'] = [False,]
        data_params_payload['position_sizing_to_trade'] = [None,]

        # These are dependent variables (that affects cost)
        data_params_payload['instrument_to_trade'] = ["USDSGD",]
        data_params_payload["backtest_window"] = [["2023-01-01","2023-12-31"]]
        data_params_payload["signal_to_trade"] = ["zscore"]
        data_params_payload["signal_function"] = ["z_sig"]
        data_params_payload["model_name"] = ["Mean_Reversion"]
        data_params_payload["tradable_times"] = [[["00:00","08:59"],],]
        data_params_payload["days_of_the_week_to_trade"] = [[0,1,2,3,4]]
        data_params_payload['timeframe_to_trade'] = ["5m"]

        data_params_payload['run_signals_w_TP'] = [False,]  

        data_params_payload['fee'] = [[0.0001,0.0001],]
        data_params_payload['slippage'] = [[0.000,0.000]]
        data_params_payload['long_equity'] = [10000,]
        data_params_payload['long_notional'] = [10000,]
        data_params_payload['short_equity'] = [10000,]
        data_params_payload['short_notional'] = [10000,]


        data_params_payload['tide_dynamic_run'] = [True,False]
        data_params_payload['tide_dynamic_alt'] = [True,False]
        data_params_payload['z_dynamic_run'] = [True,False,]
        data_params_payload['z_dynamic_alt'] = [True,False]

        data_params_payload['sig_lag'] = [[0,0],[1,0]]
        data_params_payload['reduce_only'] = [False,]
        data_params_payload['volume_to_trade'] = ["volume"]
        data_params_payload['kline_to_trade'] = ["close"]

        data_params_payload['MFI_sharpe_windows'] = [[8, 13, 21],]
        data_params_payload['MFI_strong_windows'] = [[8, 13, 21],]
        data_params_payload['MFI_weak_windows'] = [[34,45,55],]
        data_params_payload['MFI_flat_windows'] = [[89,121,144],]
        # =======================================================================================================
        # tp_position_dict
        # =======================================================================================================

        lookbacks = [8, 13,30,34,55,89]  # lookback values
        qtls_L = [[0.99, 0.95, 0.8],[0.2, 0.05, 0.01]]  # qtl values for L
        qtls_S = [[0.99, 0.95, 0.8], [0.2, 0.05, 0.01]]  # qtl values for S, different from qtls_L
        tp_position_dicts = []  # list to store all tp_position_dict

        """
                {"TP1": {"L":{"lookback":30, "qtl": 0.35}, 
                                                        "S": {"lookback":30, "qtl":0.35}
                                                        },
                                                "TP2": {"L":{"lookback":30, "qtl": 0.65}, 
                                                        "S": {"lookback":30, "qtl":0.65}
                                                        },
                                                "TP3": {"L":{"lookback":30, "qtl": 0.95}, 
                                                        "S": {"lookback":30, "qtl":0.95}
                                                        }
                                                }
        """
        # Iterate over all combinations of lookback and qtl
        # for lookback in lookbacks:
        #     for i in range(len(qtls_L)):
        #         tp_position_dict = {
        #             "TP1": {"L": {"lookback": lookback, "qtl": qtls_L[i][0]}, 
        #                     "S": {"lookback": lookback, "qtl": qtls_S[i][0]}},
        #             "TP2": {"L": {"lookback": lookback, "qtl": qtls_L[i][1]}, 
        #                     "S": {"lookback": lookback, "qtl": qtls_S[i][1]}},
        #             "TP3": {"L": {"lookback": lookback, "qtl": qtls_L[i][2]}, 
        #                     "S": {"lookback": lookback, "qtl": qtls_S[i][2]}}
        #         }
        #         tp_position_dicts.append(tp_position_dict)

        tp_position_dict =  {"TP1": {"L":{"lookback":30, "qtl": 0.35}, 
                                "S": {"lookback":30, "qtl":0.35}
                                },
                        "TP2": {"L":{"lookback":30, "qtl": 0.65}, 
                                "S": {"lookback":30, "qtl":0.65}
                                },
                        "TP3": {"L":{"lookback":30, "qtl": 0.95}, 
                                "S": {"lookback":30, "qtl":0.95}
                                }
                        }
        tp_position_dicts.append(tp_position_dict)

        data_params_payload['tp_position_dict'] = tp_position_dicts
    # =======================================================================================================
    # Feature params (continuous)
    # =======================================================================================================
        data_params_payload['SL_penalty'] = [1,]
        data_params_payload['L_buy'] = [-4,] # [-5, -4, -3, -2, -1]
        data_params_payload['L_sell'] = [-1,] # [-5, -4, -3, -2, -1]
        data_params_payload['S_buy'] = [5,4,3,2,1] # [5, 4, 3, 2, 1]
        data_params_payload['S_sell'] = [5,4,3,2,1] # [5, 4, 3, 2, 1]
        data_params_payload['min_holding_period'] = [0,]
        data_params_payload['max_holding_period'] = [60]
        data_params_payload['MFI_window'] = [60,] # [12,60,]
        data_params_payload['MFI_max_lookback'] = [80,] # [16, 80,]


        data_params_payload['MFI_sharpe_threshold'] = [5,]
        data_params_payload['MFI_sharpe_sensitivity'] = [0.5,]

        data_params_payload['MFI_sharpe_strong_level'] = [0.67,] # [0.67, 0.1, 0.9] # [0.99, 0.80, 0.05,] # [0.99,0.95, 0.80,0.2, 0.1, 0.05,]
        data_params_payload['MFI_strong_threshold'] = [5,]
        data_params_payload['MFI_strong_sensitivity'] = [0.5,]

        data_params_payload['MFI_sharpe_weak_level'] = [0.67,] # [0.67, 0.1, 0.9] # [0.05, 0.8, 0.99,] # [0.05, 0.1, 0.2, 0.8, 0.95, 0.99,]
        data_params_payload['MFI_weak_threshold'] = [7,]
        data_params_payload['MFI_weak_sensitivity'] = [0.5,]

        data_params_payload['MFI_flat_threshold'] = [10,]
        data_params_payload['MFI_flat_sensitivity'] = [0.5,]

        data_params_payload['tide_strong_level'] = [5, ] 
        data_params_payload['tide_strong_window'] = [240,] # [48, 288]
        data_params_payload['tide_strong_threshold'] = [2]
        
        data_params_payload['tide_weak_level'] = [-5, ] 
        data_params_payload['tide_weak_window'] = [240,] 
        data_params_payload['tide_weak_threshold'] = [1.5,] # [2,1]

        data_params_payload['tide_flat_window'] =   [240, ]
        data_params_payload['tide_flat_threshold'] = [2]

        if params_payload_to_add is not None:
                for k,v in params_payload_to_add.items():
                        data_params_payload[k] = v

        return data_params_payload





def organize_params(data_params_payload, use_preset_initial_values=True):
        # Predefined lists of continuous and non-continuous variables
        non_continuous_vars = ['instrument_to_trade','backtest_window','signal_to_trade',
                                'signal_function','model_name','tradable_times','days_of_the_week_to_trade',
                                'timeframe_to_trade','show_plots', 'show_plots_TP','show_rolling_SR',
                                'run_signals_w_TP', 'mutate_signals_w_TP','position_sizing_to_trade',
                                'fee', 'slippage','long_equity','long_notional','short_equity','short_notional',
                                'tide_dynamic_run','tide_dynamic_alt','z_dynamic_run','z_dynamic_alt','sig_lag',
                                'reduce_only','volume_to_trade','kline_to_trade',
                                'MFI_sharpe_windows','MFI_strong_windows','MFI_weak_windows','MFI_flat_windows','tp_position_dict'
                                ]
        
        
        gradient_descent_hyperparams = {'SL_penalty':              {"initial_value":1,"learning_rate":0.05, "epsilon":0.1, "clipping": [0,3]},
                                        
                                        'L_buy':                   {"initial_value":-1, "learning_rate":-1, "epsilon":-0.5, "clipping": [-10,10]},
                                        'L_sell':                  {"initial_value":-1, "learning_rate":-1, "epsilon":-0.5, "clipping": [-10,10]},
                                        'S_buy':                   {"initial_value":1, "learning_rate":1, "epsilon":0.5, "clipping": [-10,10]},
                                        'S_sell':                  {"initial_value":1, "learning_rate":1, "epsilon":0.5, "clipping": [-10,10]},
                                        
                                        'min_holding_period':      {"initial_value":0, "learning_rate":1, "epsilon":10, "clipping": [0,20]},
                                        'max_holding_period':      {"initial_value":60, "learning_rate":1, "epsilon":10, "clipping": [60,100]},
                                        
                                        'MFI_window':              {"initial_value":60, "learning_rate":1, "epsilon":10, "clipping": [30,100]},
                                        'MFI_max_lookback':        {"initial_value":80, "learning_rate":1, "epsilon":10, "clipping": [80,100]},
                                        
                                        'MFI_sharpe_threshold':    {"initial_value":1, "learning_rate":3, "epsilon":3, "clipping": [5,10]},
                                        'MFI_sharpe_sensitivity':  {"initial_value":0.1, "learning_rate":0.2, "epsilon":0.2, "clipping": [0.5,1]},
                                        
                                        'MFI_sharpe_strong_level': {"initial_value":0.1, "learning_rate":0.2, "epsilon":0.2, "clipping": [0.2,1]},
                                        'MFI_strong_threshold':    {"initial_value":5, "learning_rate":2, "epsilon":2, "clipping": [5,10]},
                                        'MFI_strong_sensitivity':  {"initial_value":0.5, "learning_rate":0.2, "epsilon":0.2, "clipping": [0.5,1]},
                                        
                                        'MFI_sharpe_weak_level':   {"initial_value":0.1, "learning_rate":0.2, "epsilon":0.2, "clipping": [0.2,1]},
                                        'MFI_weak_threshold':      {"initial_value":1, "learning_rate":1,  "epsilon":0.2,  "clipping": [7,10]},
                                        'MFI_weak_sensitivity':    {"initial_value":0.1, "learning_rate":0.1, "epsilon":0.2,  "clipping": [0.1,1]},

                                        'MFI_flat_threshold':      {"initial_value":2, "learning_rate":5,"epsilon":0.2, "clipping": [10,20]},
                                        'MFI_flat_sensitivity':    {"initial_value":0.1, "learning_rate":0.1, "epsilon":0.3,"clipping": [0.2,1]},

                                        'tide_strong_level':       {"initial_value":7, "learning_rate":1, "epsilon":1, "clipping": [5,10]},     
                                        'tide_strong_window':      {"initial_value":300, "learning_rate":5, "epsilon":10, "clipping": [240,400]},
                                        'tide_strong_threshold':   {"initial_value":3, "learning_rate":1, "epsilon":3, "clipping": [2,5]},     
                                        
                                        'tide_weak_level':         {"initial_value":1, "learning_rate": 1, "epsilon": 2, "clipping": [-2,10]},   
                                        'tide_weak_window':        {"initial_value":300, "learning_rate":1, "epsilon":10, "clipping": [240,400]},
                                        'tide_weak_threshold':     {"initial_value":2, "learning_rate":0.5, "epsilon":1, "clipping": [1.5,3]},
                                        
                                        'tide_flat_window':        {"initial_value":300, "learning_rate":5, "epsilon":10, "clipping": [240,400]},
                                        'tide_flat_threshold':     {"initial_value":3, "learning_rate":0.1, "epsilon":0.5, "clipping": [0,4]},
                                        }
        continuous_vars = list(gradient_descent_hyperparams.keys())
        if use_preset_initial_values:
               initial_values_dict = {var: gradient_descent_hyperparams[var]["initial_value"] for var in continuous_vars}
        else:
                initial_values_dict = {var: data_params_payload[var] for var in continuous_vars}
        learning_rates = {var: gradient_descent_hyperparams[var]["learning_rate"] for var in continuous_vars}
        epsilons = {var: gradient_descent_hyperparams[var]["epsilon"] for var in continuous_vars}
        clippings = {var: gradient_descent_hyperparams[var]["clipping"] for var in continuous_vars}
        

        continuous_variables = {var: data_params_payload[var] for var in continuous_vars}
        non_continuous_variables = {var: data_params_payload[var] for var in non_continuous_vars}

        return continuous_variables, non_continuous_variables, initial_values_dict, learning_rates, epsilons, clippings

def build_data_params_payload():
     return {'instrument_to_trade': 'USDSGD',
                'backtest_window': ['2023-01-01', '2023-12-31'],
                'signal_to_trade': 'zscore',
                'signal_function': 'z_sig',
                'model_name': 'Mean_Reversion',
                'tradable_times': [['00:00', '08:59']],
                'days_of_the_week_to_trade': [0, 1, 2, 3, 4],
                'timeframe_to_trade': '5m',
                'show_plots': False,
                'show_plots_TP': False,
                'show_rolling_SR': False,
                'run_signals_w_TP': False,
                'mutate_signals_w_TP': False,
                'position_sizing_to_trade': None,
                'fee': [0.0001, 0.0001],
                'slippage': [0.0, 0.0],
                'long_equity': 10000,
                'long_notional': 10000,
                'short_equity': 10000,
                'short_notional': 10000,
                'tide_dynamic_run': True,
                'tide_dynamic_alt': False,
                'z_dynamic_run': False,
                'z_dynamic_alt': False,
                'sig_lag': [1, 0],
                'reduce_only': False,
                'volume_to_trade': 'volume',
                'kline_to_trade': 'close',
                'MFI_sharpe_windows': [8, 13, 21],
                'MFI_strong_windows': [8, 13, 21],
                'MFI_weak_windows': [34, 45, 55],
                'MFI_flat_windows': [89, 121, 144],
                'tp_position_dict': {'TP1': {'L': {'lookback': 30, 'qtl': 0.35},
                'S': {'lookback': 30, 'qtl': 0.35}},
                'TP2': {'L': {'lookback': 30, 'qtl': 0.65},
                'S': {'lookback': 30, 'qtl': 0.65}},
                'TP3': {'L': {'lookback': 30, 'qtl': 0.95},
                'S': {'lookback': 30, 'qtl': 0.95}}},
                'SL_penalty': 1,
                'L_buy': -4,
                'L_sell': -1,
                'S_buy': 5,
                'S_sell': 2,
                'min_holding_period': 0,
                'max_holding_period': 30,
                'MFI_window': 60,
                'MFI_max_lookback': 80,
                'MFI_sharpe_threshold': 5,
                'MFI_sharpe_sensitivity': 0.5,
                'MFI_sharpe_strong_level': 0.67,
                'MFI_strong_threshold': 5,
                'MFI_strong_sensitivity': 0.5,
                'MFI_sharpe_weak_level': 0.67,
                'MFI_weak_threshold': 7,
                'MFI_weak_sensitivity': 0.5,
                'MFI_flat_threshold': 10,
                'MFI_flat_sensitivity': 0.2,
                'tide_strong_level': 1,
                'tide_strong_window': 240,
                'tide_strong_threshold': 2,
                'tide_weak_level': -1,
                'tide_weak_window': 240,
                'tide_weak_threshold': 1.5,
                'tide_flat_window': 240,
                'tide_flat_threshold': 2}