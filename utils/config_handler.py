import configparser
import os
import sys

# THIS SETS environment variable as dev to be abstracted away
os.environ["BOT_ENVIRONMENT"] = "dev"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOT_W_DIR = "./"

def get_config_path():
    """
    Read config.ini file and load into memory
    """
    
    env = os.getenv('BOT_ENVIRONMENT')
    env = env.lower()
    
    
    if env == 'dev':
        config_path = os.path.join(BOT_W_DIR, 'config', 'config-dev.ini')
    elif env == 'test':
        config_path = os.path.join(BOT_W_DIR, 'config', 'config-test.ini')
    elif env == 'prod':
        config_path = os.path.join(BOT_W_DIR, 'config', 'config-prod.ini')
        
    return config_path 

    

def get_config(section, key):
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return config.get(section, key)


def get_config_bool(section, key):
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return config.getboolean(section, key)

def get_config_all(section):
    """
    Returns a dictionary of all parameters
    """
    
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return {k:v for k,v in config[section].items()}

def write_config(section, key, value):
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    
    config_file = open(config_path, 'w')
    config.set(section, key, value)
    config.write(config_file)
    config_file.close()
    
    print(f"config file updated for section: {section}, key: {key}, value: {value}")