import re
from pathlib import Path
from typing import Any, Dict, List


import rapidjson


def log_config_error_range(path: str, errmsg: str) -> str:
    """
    Parses configuration file and prints range around error
    """
    if path != '-':
        offsetlist = re.findall(r'(?<=Parse\serror\sat\soffset\s)\d+', errmsg)
        if offsetlist:
            offset = int(offsetlist[0])
            text = Path(path).read_text()
            # Fetch an offset of 80 characters around the error line
            subtext = text[offset - min(80, offset):offset + 80]
            segments = subtext.split('\n')
            if len(segments) > 3:
                # Remove first and last lines, to avoid odd truncations
                return '\n'.join(segments[1:-1])
            else:
                return subtext
    return ''



def load_config_file(path: str) -> Dict[str,Any]:
    """
    Loads a config file from given patj
    """
    try:
        with open(path,) as json_file:
            config_dict = rapidjson.load(json_file)
            
    except FileNotFoundError:
        raise Exception(
            f'Config file "{path}" not found!'
            ' Please create a config file or check whether it exists.')
            
    except rapidjson.JSONDecodeError as e:
        err_range = log_config_error_range(path, str(e))
        raise Exception(
            f'{e}\n'
            f'Please verify the following segment of your configuration:\n{err_range}'
            if err_range else 'Please verify your configuration file for syntax errors.'
        )
        
    return config_dict
        
        
if __name__ == "__main__":
    
    path = "./config/USDSGD_1m_test.json"
    config_dict = load_config_file(path)
    print(config_dict)
    # path = "./config/config-dev.json"
    # config_dict = load_config_file(path)