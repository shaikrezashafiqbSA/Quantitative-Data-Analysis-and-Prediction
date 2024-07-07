
from main.Trading_Bot import Trading_Bot
import json

if __name__ == "__main__":


    """
        Example model_config:

        config = {
                  'instruments': ['USDSGD', 'C:USDSGD'],
                  'instrument_index_to_trade': 0,
                  'instruments_to_query': ['KGI__USDSGD', 'polygon__C:USDSGD'],
                  'timeframes_to_query': ['1m'],
                  'timeframe_to_trade': '1m',
                  'data_window': ['2023-11-01 00:00:00', '2026-12-31 00:00:00'], # Change this to reduce data loadign time too
                  'memory_len': 20000,
                  'limits': {'polygon': 5000, 'ccxt': 5000},
                  'instrument_to_trade': 'USDSGD',
                  'resample_to_list': ['1m'],
                  'update': True,
                  'stage_df_name': 'df_stage_KGI_1m',
                  'use_alt_volume': True,
                  'clean_tradable_times': [['00:00', '22:00']], # DONT CHANGE THIS. THIS IS SPECIFIC FOR KGI (since 2200 - 0000 got data anomalies)
                  'since': '2020-01-01 00:00:00',
                  'clean_base_data_flag': True,
                  'order_type': 'market',
                  'data_update': True,
                  'backtest_window': ['2021-01-01', '2023-12-31'],
                  'precision': 5,
                  'tick_size': 1e-05,
                  'run_mode': 'live',
                  'show_plots': False,
                  'show_plots_TP': False,
                  'mutate_signals_w_TP': False,
                  'show_rolling_SR': False,
                  'position_sizing_to_trade': None,
                  'signal_to_trade': 'zscore',
                  'signal_function': 'z_sig',
                  'model_name': 'Mean_Reversion',
                  'tradable_times': [['00:00', '22:00']],
                  'days_of_the_week_to_trade': [0, 1, 2, 3, 4],
                  'run_signals_w_TP': False,
                  'fee': [0.0001, 0.0001],
                  'slippage': [0.0, 0.0],
                  'long_equity': 10000,
                  'long_notional': 1000,
                  'short_equity': 10000,
                  'short_notional': 1000,
                  'tide_dynamic_run': True,
                  'tide_dynamic_alt': True,
                  'z_dynamic_run': True,
                  'z_dynamic_alt': False,
                  'sig_lag': [0, 0],
                  'reduce_only': False,
                  'volume_to_trade': 'volume',
                  'kline_to_trade': 'close',
                  'MFI_sharpe_windows': [8, 13, 21],
                  'MFI_strong_windows': [8, 13, 21],
                  'MFI_weak_windows': [34, 45, 55],
                  'MFI_flat_windows': [89, 121, 144],
                  'tp_position_dict': {
                    'TP1': {'L': {'lookback': 30, 'qtl': 0.35}, 'S': {'lookback': 30, 'qtl': 0.35}},
                    'TP2': {'L': {'lookback': 30, 'qtl': 0.65}, 'S': {'lookback': 30, 'qtl': 0.65}},
                    'TP3': {'L': {'lookback': 30, 'qtl': 0.95}, 'S': {'lookback': 30, 'qtl': 0.95}}
                  },
                  'SL_penalty': 1,
                  'L_buy': -1,
                  'L_sell': -1,
                  'S_buy': 1,
                  'S_sell': 1,
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
                  'tide_flat_threshold': 2
        }

        formatted_config = json.dumps(config, indent=4)
        print(formatted_config)
    """
    trading_bot = Trading_Bot(reload_model_state = False,
                                graceful_resume = False,
                                bot_timeframe_delay_s = 3,
                                bot_position_update_interval_m = 30,

                                prepare_plots_flag = True,
                                publish_html_plotly_dashboard = False,
                                publish_html_file_path = "./backtests/trading_bot_plotly/",
                                
                                replay = True,
                                replay_window = ["2023-12-29 00:00:00", "2024-01-01 00:00:00"], # None
                                replay_speed = 10, # None

                                model_selection = "USDSGD_2m_MR",
                                config_mode = "config-dev",
                                save_to_db_flag = "False",
                                run_mode = "paper", # paper # live
                                tradable_times = [["00:00", "09:00"]],
                            )

    trading_bot.start_bot()


