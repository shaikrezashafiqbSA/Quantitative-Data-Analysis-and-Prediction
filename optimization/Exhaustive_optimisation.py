from tqdm import tqdm
import winsound
import time
import numpy as np
from multiprocessing import Pool
import itertools

from utils.pickle_helper import pickle_this
from backtesters.get_trading_decision import run_backtest
from optimization.parameter_construction import build_params

# ============================================================================
# HyperParams
# ============================================================================
# in build_params

# ============================================================================
# Load params_payload
# ============================================================================
data_params_payloads = build_params()
params_list = [v for k,v in data_params_payloads.items()]
# Use itertools.product to generate all combinations of parameters
params_product = list(itertools.product(*params_list))
# Convert the first tuple in params_product to a dictionary
params = []
for i in range(len(params_product)):
    params.append(dict(zip(data_params_payloads.keys(), params_product[i])))  

params = params#[0:240]

def param_generator(data_params_payloads):
    params_list = [v for k,v in data_params_payloads.items()]
    for values in itertools.product(*params_list):
        yield dict(zip(data_params_payloads.keys(), values))

# print(f"={'='*50}\n{'=' + 'Beginning optimisation'.center(48,' ') + '='}\n{'='*50}")
if __name__ == '__main__':

    workers = 12
    batch_size = workers*4
    batches = [params[i:i + batch_size] for i in range(0, len(params), batch_size)]

    input_str = f"\n={'='*50}\n{'=' + 'Running Backtests'.center(48,' ') + '='}\n{'='*50}" +\
                f"\n\nNumber of parameter sets: {len(params)}\n\nWorkers:{workers}\n\nBatches: {len(batches)}\n\nclick 'y' to continue: "

    # if user type y then continue
    if input(input_str).lower() == "y":
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

        t_start = time.time()
        # ========================================================================
        t_start_parallel = time.time()
        with Pool(processes=workers) as pool:
            for i, batch in enumerate(batches):
                t_start_batch = time.time()

                try:
                    batch_results = list(tqdm(pool.imap(run_backtest, batch), total=len(batch)))
                except Exception as e:
                    print(e)
                    print("Error in batch, trying one more time...")
                    # if failed then try again:
                    batch_results = list(tqdm(pool.imap(run_backtest, batch), total=len(batch)))
                # pickle and save batch_results

                old_results = pickle_this(pickle_name=f"results", path="./optimization/parameters/")
                if old_results is None:
                    print(f"creating new ./optimization/parameters/results.pickle ...")
                    old_results = []
                old_results.extend(batch_results)
                pickle_this(data=old_results, pickle_name=f"results", path="./optimization/parameters/")
                t_end_batch = time.time()

                dur_batch = np.round((t_end_batch-t_start_batch)/60,2)
                total_eta = np.round((dur_batch)*len(batches),2)
                eta_done = np.round((dur_batch)*i,2)
                # eta_so_far = np.round((t_end_batch-t_start_parallel),3)
                print(f"{'=' + f'batch {i}/{len(batches)} ({eta_done} mins /{total_eta} mins) ---> completed in {dur_batch} mins : '.center(300,' ') + '='}")
        # ========================================================================
        print(f"\n\nTime taken: {(np.round(time.time() - t_start)/60,2)}s")
        print("Exiting...")
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    else:
        print("Exiting...")
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)