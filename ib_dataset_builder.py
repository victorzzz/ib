import multiprocessing
import logging
import pandas as pd

from typing import Optional

import ib_tickers_cache as ib_tickers_cache
import constants as cnts
import ib_tickers as ib_tckrs
import ib_logging as ib_log
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import df_tech_indicator_utils as df_tech_utils

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
            df[column].fillna(method='ffill', inplace=True)
        elif column in trades_volume_fields:
            df[column].fillna(0.0, inplace=True)

def reverse_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    result:pd.DataFrame = df.iloc[::-1]
    result.reset_index(drop=True, inplace=True)

    result = result.copy()

    return result

def add_minute_multiplier_to_column_names(df:pd.DataFrame, minute_multiplier:int) -> pd.DataFrame:
    result:pd.DataFrame = df.add_prefix(f"{minute_multiplier}m_")
    result.set_index(f"{minute_multiplier}m_timestamp", inplace=True)

    return result

def create_datasets(
        tickers:list[tuple[str, list[str]]],
        lock, 
        shared_tickers_cache:dict[str, int]):
    
    logging.info(f"Processing tickers: {tickers}")

    for ticker in tickers:
        ticker_symbvol:str = ticker[0]
        ticker_exchanges:list[str] = ticker[1]
        
        for exchange in ticker_exchanges:

            logging.info(f"Loading merged dataframes '{ticker_symbvol}' - '{exchange}' ...")

            dfs = load_merged_dataframes(ticker_symbvol, exchange, lock, shared_tickers_cache)

            enriched_dfs = {}

            for minute_multiplier, df in dfs.items():

                logging.info(f"Processing '{ticker_symbvol}' - '{exchange}' - {minute_multiplier} ...")

                logging.info(f"Replasing nan values ...")

                replace_nan_values(df)
                
                logging.info(f"Adding normalized time columns ...")

                df = df_dt_utils.add_normalized_time_columns(df)

                logging.info(f"Reversing dataframe ...")

                df = reverse_dataframe(df)

                logging.info(f"Adding technical indicators ...")

                df = df_tech_utils.add_technical_indicators(df)

                logging.info(f"Adding volume profile ...")

                df = df_tech_utils.add_volume_profile(df)
                
                logging.info(f"Adding minute multiplier to column names ...")

                df = add_minute_multiplier_to_column_names(df, minute_multiplier)

                enriched_dfs[minute_multiplier] = df

                logging.info(f"Saving dataset ...")
                
                result_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset.csv"
                df.to_csv(result_file_name, index=False)

            dfs.clear()


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