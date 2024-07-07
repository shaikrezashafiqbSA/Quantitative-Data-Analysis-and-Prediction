import os 
import sys
import re

import logging
from datetime import datetime

from logging.handlers import TimedRotatingFileHandler 
from config.load_config import load_config_file


def setup_logger(logger_file_name="test"):
    if not os.path.exists(f'./logs'):
        os.mkdir(f'./logs')
        
    filename = datetime.now().strftime(f"./logs/{logger_file_name}_%d_%b_%Y_%H_%M.log")
    
    # Get the logger
    logger = logging.getLogger(logger_file_name)
    logger.setLevel(logging.INFO)
    
    # Check if the logger already has handlers
    if not logger.handlers:
        formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(threadName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        # define rotating file handler (rotate every midnight)
        fh = TimedRotatingFileHandler(filename, when="midnight", interval=1, backupCount=365, utc=True)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # define console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # Prevent the log messages from propagating to the root logger
    logger.propagate = False
    
    return logger

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    data_dict = {}
    for line in lines:
        if "data_params_payloads_i:" in line:
            data_dict["data_params_payloads_i"] = int(re.search(r'\d+', line).group())
        elif "Iteration" in line:
            data_dict["iteration"] = int(re.search(r'\d+', line).group())
        elif "costs: total:" in line:
            costs = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            data_dict["total_cost"] = float(costs[0])
            data_dict["long_cost"] = float(costs[1])
            data_dict["short_cost"] = float(costs[2])
        elif "ORIGINAL:" in line:
            params = re.findall(r"'(\w+)': ([-+]?\d*\.\d+|\d+)", line)
            for param, value in params:
                data_dict[param] = float(value)
        elif "AFTER UPDATE:" in line:
            params = re.findall(r"'(\w+)': ([-+]?\d*\.\d+|\d+)", line)
            for param, value in params:
                data_dict[param+"_updated"] = float(value)
            data.append(data_dict)
            data_dict = {}

    return data
    


def test_logger():
    logger = setup_logger()
    logger.info("SIGNAL TRIGGER")
    logger.error("NO ORDER RESPONSE")
    logger.warning("model state not found")
    logger.info("runtime completed")
    
if __name__ == "__main__":
    test_logger()