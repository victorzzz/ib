import multiprocessing
import logging
import pandas as pd

import ta
import ta.trend
import ta.momentum
import ta.volatility
import ta.volume

from typing import Optional

import ib_tickers_cache as ib_tickers_cache
import constants as cnts
import ib_tickers as ib_tckrs
import ib_logging as ib_log
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils

midpoint_fields = ["MIDPOINT_open", "MIDPOINT_high", "MIDPOINT_low", "MIDPOINT_close"]
bid_fields = ["BID_open", "BID_high", "BID_low", "BID_close"]
ask_fields = ["ASK_open", "ASK_high", "ASK_low", "ASK_close"]
oiv_fields = ["OPTION_IMPLIED_VOLATILITY_open", "OPTION_IMPLIED_VOLATILITY_high", "OPTION_IMPLIED_VOLATILITY_low", "OPTION_IMPLIED_VOLATILITY_close"]
trades_price_fields = ["TRADES_open", "TRADES_high", "TRADES_low", "TRADES_close", "TRADES_average"]
trades_volume_fields = ["TRADES_volume", "TRADES_barCount"]

def load_merged_dataframes(
        ticker_symbvol:str, 
        exchange:str,
        lock, 
        shared_tickers_cache:dict[str, int]) -> dict[int, pd.DataFrame]:
    
    merged_dataframes = {}

    contract_id:Optional[int] = ib_tickers_cache.get_contact_id(ticker_symbvol, exchange, lock, shared_tickers_cache)

    for minute_multiplier in cnts.minute_multipliers:
        merged_file_name = f"{cnts.merged_data_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged.csv"
        if fsu.is_file_exists(merged_file_name):
            merged_dataframes[int(minute_multiplier)] = pd.read_csv(merged_file_name)
        else:
            logging.error(f"File '{merged_file_name}' does not exist")

    return merged_dataframes

def replace_nan_values(df:pd.DataFrame):
    for column in df.columns:
        if column in midpoint_fields or column in bid_fields or column in ask_fields or column in trades_price_fields:
            df[column].fillna(method='bfill', inplace=True)
        elif column in trades_volume_fields:
            df[column].fillna(0.0, inplace=True)

def add_technical_indicators(df:pd.DataFrame) -> pd.DataFrame:

    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)

    df['ADI_MIDPOINT'] = ta.volume.acc_dist_index(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'])
    df['ADI_TRADES_average'] = ta.volume.acc_dist_index(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'])

    df['OBV_MIDPOINT'] = ta.volume.on_balance_volume(df['MIDPOINT_close'], df['TRADES_volume'])
    df['OBV_TRADES_average'] = ta.volume.on_balance_volume(df['TRADES_average'], df['TRADES_volume'])

    for period in (10, 20, 30, 40):
        df[f'CMF_MIDPOINT_{period}'] = ta.volume.chaikin_money_flow(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], window=period)
        df[f'CMF_TRADES_average_{period}'] = ta.volume.chaikin_money_flow(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], window=period)

    for period in (6, 13, 26):
        df[f'FI_MIDPOINT_{period}'] = ta.volume.force_index(df['MIDPOINT_close'], df['TRADES_volume'], window=period)
        df[f'FI_TRADES_average_{period}'] = ta.volume.force_index(df['TRADES_average'], df['TRADES_volume'], window=period)

    df['VPT_MIDPOINT'] = ta.volume.volume_price_trend(df['MIDPOINT_close'], df['TRADES_volume'])
    df['VPT_TRADES_average'] = ta.volume.volume_price_trend(df['TRADES_average'], df['TRADES_volume'])

    df['NVI_MIDPOINT'] = ta.volume.negative_volume_index(df['MIDPOINT_close'], df['TRADES_volume'])
    df['NVI_TRADES_average'] = ta.volume.negative_volume_index(df['TRADES_average'], df['TRADES_volume'])

    for period in (7, 14, 21, 28):
        df[f'EOM_MIDPOINT_{period}'] = ta.volume.ease_of_movement(df['MIDPOINT_high'], df['MIDPOINT_low'], df['TRADES_volume'], window=period)
        df[f'EOM_TRADES_{period}'] = ta.volume.ease_of_movement(df['TRADES_high'], df['TRADES_low'], df['TRADES_volume'], window=period)

    for period in (7, 14, 21, 28):
        df[f'ATR_MIDPOINT_{period}'] = ta.volatility.average_true_range(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], window=period)
        df[f'ATR_TRADES_average_{period}'] = ta.volatility.average_true_range(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], window=period)

    for period in (7, 14, 21, 28):
        df[f'RSI_MIDPOINT_{period}'] = ta.momentum.rsi(df['MIDPOINT_close'], window=period)
        df[f'RSI_TRADES_average_{period}'] = ta.momentum.rsi(df['TRADES_average'], window=period)

    for period in (10, 20, 40):
        df[f'BB_MAVG_MIDPOINT_{period}'] = ta.volatility.bollinger_mavg(df['MIDPOINT_close'], window=period)
        df[f'BB_MAVG_TRADES_average_{period}'] = ta.volatility.bollinger_mavg(df['TRADES_average'], window=period)
        for k in (2, 3):
            df[f'BB_HBAND_MIDPOINT_{period}_{k}'] = ta.volatility.bollinger_hband(df['MIDPOINT_close'], window=period, window_dev=k)
            df[f'BB_HBAND_TRADES_average_{period}_{k}'] = ta.volatility.bollinger_hband(df['TRADES_average'], window=period, window_dev=k)
            df[f'BB_LBAND_MIDPOINT_{period}_{k}'] = ta.volatility.bollinger_lband(df['MIDPOINT_close'], window=period, window_dev=k)
            df[f'BB_LBAND_TRADES_average_{period}_{k}'] = ta.volatility.bollinger_lband(df['TRADES_average'], window=period, window_dev=k)

    for period, s in ((7,2), (10,2), (14,3), (21,4), (28,6)):
        df[f'STOCH_MIDPOINT_{period}_{s}'] = ta.momentum.stoch(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], window=period, smooth_window=s)
        df[f'STOCH_TRADES_average_{period}_{s}'] = ta.momentum.stoch(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], window=period, smooth_window=s)

    for step in (0.01, 0.02, 0.03):
        for max_step in (0.1, 0.2, 0.3):
            df[f'PSAR_MIDPOINT_{step}_{max_step}'] = ta.trend.psar_up(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], step=step, max_step=max_step)
            df[f'PSAR_TRADES_average_{step}_{max_step}'] = ta.trend.psar_up(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], step=step, max_step=max_step)
            df[f'PSAR_MIDPOINT_{step}_{max_step}'] = ta.trend.psar_down(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], step=step, max_step=max_step)
            df[f'PSAR_TRADES_average_{step}_{max_step}'] = ta.trend.psar_down(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], step=step, max_step=max_step)

    for slow, fast, signal in ((20, 9, 7), (26, 12, 9), (39, 18, 13),  (52, 24, 18)):
            df[f'MACD_MIDPOINT_{slow}_{fast}'] = ta.trend.macd(df['MIDPOINT_close'], window_slow=slow, window_fast=fast)
            df[f'MACD_TRADES_average_{slow}_{fast}'] = ta.trend.macd(df['TRADES_average'], window_slow=slow, window_fast=fast)

            df[f'MACD_DIFF_MIDPOINT_{slow}_{fast}_{signal}'] = ta.trend.macd_diff(df['MIDPOINT_close'], window_slow=slow, window_fast=fast, window_sign=signal)
            df[f'MACD_DIFF_TRADES_average_{slow}_{fast}_{signal}'] = ta.trend.macd_diff(df['TRADES_average'], window_slow=slow, window_fast=fast, window_sign=signal)

            df[f'MACD_SIGNAL_MIDPOINT_{slow}_{fast}'] = ta.trend.macd_signal(df['MIDPOINT_close'], window_slow=slow, window_fast=fast)
            df[f'MACD_SIGNAL_TRADES_average_{slow}_{fast}'] = ta.trend.macd_signal(df['TRADES_average'], window_slow=slow, window_fast=fast)

    return df

def create_datasets(
        tickers:list[tuple[str, list[str]]],
        lock, 
        shared_tickers_cache:dict[str, int]):
    
    for ticker in tickers:
        ticker_symbvol:str = ticker[0]
        ticker_exchanges:list[str] = ticker[1]
        
        for exchange in ticker_exchanges:
            dfs = load_merged_dataframes(ticker_symbvol, exchange, lock, shared_tickers_cache)

            for minute_multiplier, df in dfs.items():
                replace_nan_values(df)
                df_dt_utils.add_normalized_time_columns(df)
                add_technical_indicators(df)
                

def do_step():
    processes = []

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    for tikers_batch in ib_tckrs.get_selected_tickers_batches(cnts.complex_processing_batch_size):
        processed_ticker_symbols = [ticker[0] for ticker in tikers_batch]
               
        logging.info("-------------------------------------")
        logging.info(f"Group '{', '.join(processed_ticker_symbols)}' ...")
        logging.info("-------------------------------------")

        process = multiprocessing.Process(
            target=create_datasets, 
            args=(tikers_batch, lock, shared_dict))
        
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

# ----------------------------

if __name__ == "__main__":
    
    ib_log.configure_logging("ib_raw_data_merger")

    logging.info(f"Starting {__file__} ...")

    fsu.create_required_folders()

    do_step()