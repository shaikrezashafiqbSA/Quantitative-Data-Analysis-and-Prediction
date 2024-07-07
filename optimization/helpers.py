from multiprocessing import Pool
from backtesters.get_trading_decision import run_backtest

def worker_func(batch):
    batch_results = []
    # run loops and get results for the current batch
    for param in batch:
        result = run_backtest(param)
        batch_results.append(result)
    return batch_results