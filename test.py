from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools
import os

from logger.logger import setup_logger
from utils.pickle_helper import pickle_this
from backtesters.get_trading_decision import run_backtest
from optimization.parameter_construction import organize_params, build_params
from copy import deepcopy



logger = setup_logger(logger_file_name="hybrid_optimization")

def batch(iterable, n=1):
    iterable = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterable, n))
        if not chunk:
            return
        yield chunk

def param_generator(data_params_payloads):
    params_list = [v for k,v in data_params_payloads.items()]
    for values in itertools.product(*params_list):
        yield dict(zip(data_params_payloads.keys(), values))

def two_layer_optimization(patience = 5,
                    metric_min_delta = 0.001,
                    worker_count=12, 
                    gradient_descent_max_iterations =10,
                    global_minimum = False,
                    batch_params_file_name = "cost_data_params_payload_dict"
                    ):
    optima_metric=os.getenv('optima_metric')
    logger.info(f"\n\noptima_metric: {optima_metric}\n\n")
    data_params_payloads = build_params()
    continuous_variables, non_continuous_variables, continuous_initial_values_dict, learning_rates, epsilons = organize_params(data_params_payloads)
    # Initialize the best parameters and the best cost
    best_params = None
    best_cost = -np.inf

    len_param_sets = len(data_params_payloads)
    t_start = time.time()
    # First layer: exhaustive search over the non-continuous variables
    for i, data_params_payload in tqdm(enumerate(param_generator(data_params_payloads)), desc= f"Exhaustive search iterations i"):
        non_continuous_data_params_payload = {k:v for k,v in data_params_payload.items() if k in non_continuous_variables}
        t_start_batch = time.time()
        data_params_payloads_i = i
        logger.info(f"\n{'=='*100}\nbatch {i+1} ---> \n{non_continuous_data_params_payload}\n{'=='*100}\n\n")
        # Evaluate the cost function for the current parameters
        data_params_payload1, cost = evaluate_cost(data_params_payloads_i,
                                                   data_params_payload, 
                                    non_continuous_variables=non_continuous_variables,
                                    continuous_variables=continuous_variables,
                                    continuous_initial_values_dict=continuous_initial_values_dict,
                                    learning_rates=learning_rates, 
                                    epsilons=epsilons,
                                    gradient_descent_max_iterations =gradient_descent_max_iterations ,
                                    patience = patience,
                                    metric_min_delta = metric_min_delta,
                                    optima_metric=optima_metric,
                                    worker_count=worker_count,
                                    global_minimum=global_minimum)

        # If this cost is better than the best cost, update the best parameters and the best cost
        batch_results = [{"i":i, "cost":cost, "data_params_payload":data_params_payload1}]
        # pickle optima params
        old_results = pickle_this(pickle_name=batch_params_file_name, path="./optimization/parameters/")
        if old_results is None:
            logger.info(f"creating new ./optimization/parameters/{batch_params_file_name}.pickle ...")
            old_results = []
        old_results.extend(batch_results)
        
        pickle_this(data=old_results, pickle_name=batch_params_file_name, path="./optimization/parameters/")

        if cost > best_cost:
            logger.info(f"Model Gain: cost ({cost}) - best_cost ({best_cost}) = {cost - best_cost}")
            logger.info(f"\n\nLOSER data_params_payload --> cost: {best_cost}\n {best_params}")
            logger.info(f"WINNER data_params_payload --> cost: {cost}\n {data_params_payload1}\n\n")
            best_params = data_params_payload1
            best_cost = cost

        t_end_batch = time.time()
        dur_batch = np.round((t_end_batch-t_start_batch)/60,2)
        dur_since_start = np.round((t_end_batch-t_start)/60,2)
        eta_done = np.round((dur_batch)*60/60,2)
        logger.info(f"\n{'=' + f'batch {data_params_payloads_i+1}/{len_param_sets} ({dur_since_start} mins /{eta_done} hrs) ---> completed in {dur_batch} mins : '.center(300,' ') + '='}")

    return best_params

def evaluate_cost(data_params_payloads_i,
                  data_params_payload, 
                  non_continuous_variables,
                  continuous_variables,
                  continuous_initial_values_dict, 
                  learning_rates, 
                  epsilons,
                  gradient_descent_max_iterations ,
                    patience = 5,
                    metric_min_delta = 0.001,
                    optima_metric="short",
                    worker_count=12,
                    global_minimum=False,
                    verbose=False):
    # Initialize the values of the continuous variables
    continuous_values_dict = continuous_initial_values_dict.copy()

    # Second layer: gradient descent over the continuous variables
    OPTIMIZED_continuous_values_dict = gradient_descent(data_params_payloads_i,
                                                        data_params_payload, 
                                        continuous_values_dict, 
                                        learning_rates, 
                                        epsilons,
                                        gradient_descent_max_iterations ,
                                        patience = patience,
                                        metric_min_delta = metric_min_delta,
                                        optima_metric=optima_metric,
                                        worker_count=worker_count,
                                        global_minimum=global_minimum,
                                        verbose=verbose)

    # Update the parameters with the optimized values
    data_params_payload.update(OPTIMIZED_continuous_values_dict)
    
    # Evaluate the cost function
    cost_config_dict = run_backtest(data_params_payload)
    cost = cost_config_dict[optima_metric]

    return data_params_payload, cost


def gradient_descent(data_params_payloads_i,
                     data_params_payload, 
                     continuous_values_dict, 
                     learning_rates, 
                     epsilons,
                     gradient_descent_max_iterations ,
                     patience = 5,
                     metric_min_delta = 0.001,
                     optima_metric="short",
                     worker_count=12, 
                     global_minimum=True,
                     verbose=False,
                     verbose_debug = False):
    # Initialize the best cost and a counter for non-improving iterations
    best_cost = -np.inf
    no_improvement_counter = 0
    epsilon_multiplier_str = os.getenv('epsilon_multiplier')
    epsilon_multiplier = float(epsilon_multiplier_str)

    original_epsilons = deepcopy(epsilons)
    # Initialize the values of the continuous variables
    if verbose: logger.info(f"\n\nBeginning gradient descent with initial values: \n{continuous_values_dict}\n\n")
    continuous_values_dict_descent_to_test = deepcopy(continuous_values_dict)
    
    
    for i in tqdm(range(gradient_descent_max_iterations ), desc = f"Gradient descent iterations i/n"):
        # Compute the gradient
        # logger.info(f"\niteration: {i} --> computing gradient ...\n")
        continuous_params_gradient = compute_gradient(data_params_payload, continuous_values_dict_descent_to_test, epsilons,worker_count)


        if verbose: logger.info(f"\n\n{'=='*100}\ndata_params_payloads_i:> {data_params_payloads_i}--> \n        iteration: {i+1} --> continuous_params_gradient: \n{continuous_params_gradient}\n")
        
        # ========================================================================
        # ========================================================================
        # Update the values param_value = param_value - learning_rate * gradient
        # ========================================================================
        # ========================================================================
        continuous_values_dict_descent_updated = deepcopy(continuous_values_dict_descent_to_test)
        if verbose: logger.info(f"{'__'*100}\nIteration {i+1}/{gradient_descent_max_iterations } BEFORE UPDATE:\n\n{continuous_values_dict_descent_updated}")
        for param_name in continuous_values_dict_descent_updated.keys():
            # local minimum
            if not global_minimum:
                updated_param_value = continuous_values_dict_descent_updated[param_name] + \
                                        learning_rates[param_name] * continuous_params_gradient[param_name]["gradient"]
            # Global minimum
            else:
                cost_1 = continuous_params_gradient[param_name]["cost_1"]
                cost_0 = continuous_params_gradient[param_name]["cost_0"]
                if cost_1 > cost_0:
                    better_param = continuous_params_gradient[param_name]["param_1"]
                else:
                    better_param = continuous_params_gradient[param_name]["param_0"]

                updated_param_value = better_param + \
                                        learning_rates[param_name] * continuous_params_gradient[param_name]["gradient"]
            
            
            continuous_values_dict_descent_updated[param_name] = updated_param_value
            # if param_name == "S_sell":
            #     if verbose_debug: logger.info(f"\n\n (1) param_name {param_name}- continuous_values_dict_descent_updated[{param_name}]: {continuous_values_dict_descent_updated[param_name]}")
            # if param_name == "S_sell":
            #     if verbose_debug: logger.info(f"\n\n (2)(a) continuous_values_dict_descent_to_test['S_sell'] + updated_param_value =  {continuous_values_dict_descent_updated['S_sell']} + {updated_param_value}")
            # if param_name == "S_sell":
            #     if verbose_debug: logger.info(f"\n\n (2)(b) continuous_values_dict_descent_to_test['S_sell'] {continuous_values_dict_descent_updated['S_sell']}")
            # Apply learning rate decay
            # learning_rates[param_name] *= (1.0 / (1.0 + decay_rate * i))

        if verbose_debug: logger.info(f"\n\n (3)(a) continuous_values_dict_descent_to_test['S_sell'] {continuous_values_dict_descent_updated['S_sell']}")
        # Run backtest here to see whats the cost
        # make a deep copy of data_params_payload and update it with the new values
        # ========================================================================
        # ========================================================================
        params_to_test = deepcopy(data_params_payload)
        if verbose_debug: logger.info(f"\n\n (3)(a) continuous_values_dict_descent_to_test['S_sell']: {continuous_values_dict_descent_updated['S_sell']}")
        if verbose_debug: logger.info(f"\n\n (3)(b): continuous_values_dict_descent_to_test:\n{continuous_values_dict_descent_updated}\nparams_to_test:\n{params_to_test}")
        params_to_test.update(continuous_values_dict_descent_updated)
        if verbose_debug: logger.info(f"\n\n (4) params_to_test: S_sell: {params_to_test['S_sell']}")
        if verbose_debug: logger.info(f"\n\n (5) running backtest with params S_sell:\n{continuous_values_dict_descent_updated['S_sell']}")
        cost_config_dict = run_backtest(params_to_test)
        cost = cost_config_dict[optima_metric]


    

        # ========================================================================
        # logger matter 
        costs = f"total: {cost_config_dict['total']} | long: {cost_config_dict['long']} | short: {cost_config_dict['short']}"
        continuous_params_gradient_df = pd.DataFrame(continuous_params_gradient).T
        continuous_params_gradient_df =continuous_params_gradient_df[["param_0","epsilon","param_1","cost_0","cost_1","gradient",]]
        continuous_params_gradient_df = continuous_params_gradient_df.sort_values(by=["cost_1","cost_0"], ascending=False,)
        logger.info(f"\n\n{'__'*100}\n\nbatch: {data_params_payloads_i+1}\nIteration {i+1}/{gradient_descent_max_iterations}\n\ncontinuous_params_gradient_df:\n{continuous_params_gradient_df}\n\nORIGINAL:\n{continuous_values_dict}\n\nAFTER UPDATE:\n\n{continuous_values_dict_descent_updated}\n\n{'__'*100}\ncosts: {costs} \n{'__'*100}\n\n") 
            
        if verbose_debug: 
            logger.info(f"\n\n6) Iteration {i+1} --> total: {cost_config_dict['total']} | long: {cost_config_dict['long']} | short: {cost_config_dict['short']}")
        
        # Check if the cost has improved enough
        if cost - best_cost > metric_min_delta:
            no_improvement_counter = 0
            continuous_values_dict_descent_to_test = deepcopy(continuous_values_dict_descent_updated)
            best_cost = cost
            # If there's improvement, revert the epsilon values back to their original values
            epsilons = deepcopy(original_epsilons)
        else:
            no_improvement_counter += 1
            # If there's no improvement, increase the epsilon values
            for param in epsilons.keys():
                epsilons[param] *= (1 + (epsilon_multiplier * no_improvement_counter))
            logger.warning(f"\n\nNo significant improvement for {no_improvement_counter} counts. due to cost ({cost}) - best_cost ({best_cost}) < {metric_min_delta} \n\n")

        if no_improvement_counter >= patience:
            logger.warning(f"\n\nNo significant improvement for {patience} counts. Stopping early. due to cost ({cost}) - best_cost ({best_cost}) < {metric_min_delta} \n\n")
            break

    return continuous_values_dict_descent_to_test




def compute_gradient(data_params_payload, 
                     continuous_params_dict, # NOTE: THIS IS A LIST??!
                     epsilons,
                     worker_count=12,
                     verbose=False):
    # Create a pool of worker processes
    with Pool(processes=worker_count) as pool:
        # Use a list comprehension to create a list of tasks
        tasks = [(param, data_params_payload, epsilons) for param in continuous_params_dict.keys()]
        # Use the pool's map function to compute the gradients in parallel
        gradient_payload_list = pool.starmap(compute_gradient_for_param, tasks)
    if verbose: logger.info(f"\n\ngradient_payload_list:\n{gradient_payload_list}\n\n")
    # Convert the list of results into a dictionary
    continuous_params_gradient = {item['param']: {k: v for k, v in item.items() if k != 'param'} for item in gradient_payload_list}
    if verbose: logger.info(f"\n\ncontinuous_params_gradient:\n{continuous_params_gradient}\n\n")
    return continuous_params_gradient


def compute_gradient_for_param(param, 
                               data_params_payload, 
                               epsilons,
                                verbose=False,):
    # Compute the cost at the current parameters
    # print(f"data_params_payload:\n{data_params_payload}\n\n")
    optima_metric = os.getenv('optima_metric')
    # print(f"param:\n{param}\n\n")
    original_cost_config_dict = run_backtest(data_params_payload)
    # print(f"original_cost_config_dict:\n{original_cost_config_dict}\noptima_metric: {optima_metric}\n")
    original_cost = original_cost_config_dict[optima_metric]
    # print(f"original_cost:\n{original_cost}\n\n")
    
    # Compute the cost at the parameters plus a small epsilon
    param_original = data_params_payload[param]
    data_params_payload[param] += epsilons[param]
    increased_cost_config_dict = run_backtest(data_params_payload)
    increased_cost = increased_cost_config_dict[optima_metric] # total long short 
    
    # Compute the gradient as the change in cost divided by epsilon
    gradient = (increased_cost - original_cost) / epsilons[param]
    if verbose:
        update_str = f"param_value      cost\n      {param_original}          {original_cost}\n      {data_params_payload[param]}          {increased_cost}"
        logger.info(f"\n{'__'*30}\nparam: {param}\n{update_str}\ngradient = {gradient}")
    # Reset the parameter
    updated_param = data_params_payload[param]
    data_params_payload[param] -= epsilons[param]

    gradient_payload = {"gradient":gradient,
                         "param_0":param_original,
                          "cost_0":original_cost,
                          "param_1": updated_param,
                          "cost_1":increased_cost,
                          "param":param,
                          "epsilon":epsilons[param],}

    return gradient_payload


def compute_gradient_for_param(param, 
                               data_params_payload, 
                               epsilons,
                               verbose=False,
                               ):
    optima_metric = os.getenv('optima_metric')
    use_central_difference_str = os.getenv('use_central_difference')
    use_central_difference = use_central_difference_str == 'True'


    original_cost_config_dict = run_backtest(data_params_payload)
    original_cost = original_cost_config_dict[optima_metric]
    
    param_original = data_params_payload[param]
    
    # ========================================================================
    # central difference
    # ========================================================================
    if use_central_difference:
        # Compute the cost at the parameters plus a small epsilon
        data_params_payload[param] += epsilons[param]
        increased_cost_config_dict = run_backtest(data_params_payload)
        increased_cost = increased_cost_config_dict[optima_metric]
        
        # Compute the cost at the parameters minus a small epsilon
        data_params_payload[param] -= 2 * epsilons[param]
        decreased_cost_config_dict = run_backtest(data_params_payload)
        decreased_cost = decreased_cost_config_dict[optima_metric]
        
        # Compute the gradient as the change in cost divided by epsilon
        gradient = (increased_cost - decreased_cost) / (2 * epsilons[param])

        # Reset the parameter
        param_1 = data_params_payload[param]
        data_params_payload[param] -= epsilons[param]

        gradient_payload = {"gradient":gradient,
                         "param_0":param_original,
                          "cost_0":decreased_cost,
                          "param_1": param_1,
                          "cost_1":increased_cost,
                          "param":param,
                          "epsilon":epsilons[param],}

    # ========================================================================
    # non-central difference
    # ========================================================================
    else:
        # Compute the cost at the parameters plus a small epsilon
        data_params_payload[param] += epsilons[param]
        increased_cost_config_dict = run_backtest(data_params_payload)
        increased_cost = increased_cost_config_dict[optima_metric]
        
        # Compute the gradient as the change in cost divided by epsilon
        gradient = (increased_cost - original_cost) / epsilons[param]

        # Reset the parameter
        param_1 = data_params_payload[param]
        data_params_payload[param] -= epsilons[param]

        gradient_payload = {"gradient":gradient,
                         "param_0":param_original,
                          "cost_0":original_cost,
                          "param_1": param_1,
                          "cost_1":increased_cost,
                          "param":param,
                          "epsilon":epsilons[param],}
        
    if verbose:
        update_str = f"param_value      cost\n      {param_original}          {original_cost}\n      {data_params_payload[param]}          {increased_cost}"
        logger.info(f"\n{'__'*30}\nparam: {param}\n{update_str}\ngradient = {gradient}")


    return gradient_payload


