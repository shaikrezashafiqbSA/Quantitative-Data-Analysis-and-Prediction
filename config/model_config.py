from datetime import datetime, timedelta

class model_config:
    def __init__(self, instruments, timeframes, i_fee, j_fee, price_range, tick_size, precision, order_type, memory_len):
        self.instruments = instruments
        self.timeframes = timeframes
        self.i_fee = i_fee
        self.j_fee = j_fee
        self.price_range = price_range
        self.tick_size = tick_size
        self.precision = precision
        self.order_type = order_type
        self.memory_len = memory_len
        
        self.since = (datetime.now() - timedelta(minutes=self.memory_len)).strftime('%Y-%m-%d %H:%M:%S')
        self.run_signals_w_TP = True
        self.model_name = "Mean_Reversion"
        self.MFI_window = 60
        self.instrument_index_to_trade = 0
        self.instrument_to_trade = self.instruments[self.instrument_index_to_trade]
        self.timeframe_to_trade = "1m"
        self.df = None
        self.signal_to_trade = "zscore"
        self.tradable_times = [["00:00","09:00"]]
        self.days_of_the_week_to_trade = [0,1,2,3,4]
        
        # Order managers
        self.tick_size = self.tick_size/10
        self.precision = self.precision+1
