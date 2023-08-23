import pickle

def pickle_this(data=None, pickle_name="df", path="./database/instruments/"):
    w = ['rb' if data is None else 'wb'][0]
    if "/" in pickle_name:
        pickle_name = pickle_name.replace("/","'")
    if ":" in pickle_name:
        pickle_name = pickle_name.replace(":",";")
    try:
        with open(f'{path}{pickle_name}.pickle', w) as handle:
            if data is None:
                data = pickle.load(handle)
                return data
            else: 
                pickle.dump(data, handle)
    except Exception as e:
        print(e)
        
        
#%%
if __name__ == "__main__":
    pickle_name = "ccxt_okx__BTC/USDT:USDT_1h"
    if "/" in pickle_name:
        pickle_name = pickle_name.replace("/","")
    print(pickle_name)
