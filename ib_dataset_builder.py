import multiprocessing
import logging
import pandas as pd

import np_utils as np_utils
import numpy as np
import ib_tickers_cache as ib_tickers_cache
import constants as cnts
import ib_tickers as ib_tckrs
import ib_logging as ib_log
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import df_tech_indicator_utils as df_tech_utils
import df_volume_profile_utils as df_vp_utils
import df_price_analizer as df_pa
import df_loader_saver as df_ls

midpoint_fields = ["MIDPOINT_open", "MIDPOINT_high", "MIDPOINT_low", "MIDPOINT_close"]
bid_fields = ["BID_open", "BID_high", "BID_low", "BID_close"]
ask_fields = ["ASK_open", "ASK_high", "ASK_low", "ASK_close"]
# oiv_fields = ["OPTION_IMPLIED_VOLATILITY_open", "OPTION_IMPLIED_VOLATILITY_high", "OPTION_IMPLIED_VOLATILITY_low", "OPTION_IMPLIED_VOLATILITY_close"]
trades_price_fields = ["TRADES_open", "TRADES_high", "TRADES_low", "TRADES_close", "TRADES_average"]
trades_volume_fields = ["TRADES_volume", "TRADES_barCount"]

all_price_fields = midpoint_fields + bid_fields + ask_fields + trades_price_fields
all_price_volume_fields = midpoint_fields + bid_fields + ask_fields + trades_price_fields + trades_volume_fields

PRICE_MISPRINTS_RATIO_THRESHOLD = 4.0

PRICES_DEPTH = 60
TECH_INDICATORS_DEPTH = 15
VOLUME_PROFILE_DEPTH = 5

TEST_MULTYRANGE_DATASET_SIZE = 50000

def load_merged_dataframe(multiplier:float, ticker_symbvol:str, contract_id:int, exchange:str) -> pd.DataFrame | None:
    
    merged_file_name = f"{cnts.merged_data_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{multiplier:.0f}--minute--merged"
    
    if df_ls.is_df_exists(merged_file_name):
        result = df_ls.load_df(merged_file_name)
    else:
        logging.error(f"File '{merged_file_name}' does not exist")
        result = None

    return result

def load_merged_dataframes(
        ticker_symbvol:str, 
        exchange:str,
        lock, 
        shared_tickers_cache:dict[str, int],
        multiplier: float | None = None) -> dict[int, pd.DataFrame]:
    
    merged_dataframes = {}

    contract_id:int | None = ib_tickers_cache.get_contact_id(ticker_symbvol, exchange, lock, shared_tickers_cache)

    if contract_id is None:
        logging.error(f"Contract ID for '{ticker_symbvol}' - '{exchange}' not found")
        return merged_dataframes;

    if multiplier is None:
        for minute_multiplier in cnts.minute_multipliers:
            df: pd.DataFrame | None = load_merged_dataframe(minute_multiplier, ticker_symbvol, contract_id, exchange)
            if df is not None:
                merged_dataframes[int(minute_multiplier)] = df
    else:
        df: pd.DataFrame | None = load_merged_dataframe(multiplier, ticker_symbvol, contract_id, exchange)
        if df is not None:
            merged_dataframes[int(multiplier)] = df
            
    return merged_dataframes

def process_empty_values(df:pd.DataFrame) -> pd.DataFrame:
    
    df = df.dropna(subset=all_price_volume_fields, how='any')
    df = df[(df['TRADES_volume'] > 0.0)]
   
    return df

def fix_trading_price_misprints(df:pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    
    for column in all_price_fields:
        if column not in df_copy.columns:
            continue
        
        column_data = df_copy[column]
        column_data_shifted = column_data.shift(-1, fill_value=np.nan)
        ratio_change = column_data / column_data_shifted
        misprints = (ratio_change > PRICE_MISPRINTS_RATIO_THRESHOLD)
        
        df_copy.loc[misprints, column] = column_data_shifted[misprints]
    
    return df_copy

def add_minute_multiplier_to_column_names(df:pd.DataFrame, minute_multiplier:int) -> pd.DataFrame:
    result:pd.DataFrame = df.add_prefix(f"{minute_multiplier}m_")

    return result

def reverse_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    result:pd.DataFrame = df.iloc[::-1]
    result.reset_index(drop=True, inplace=True)

    result = result.copy()

    return result

"""
def create_test_multyrange_dataset(enriched_dfs:dict[int, pd.DataFrame], size:int) -> None:
    main_df = enriched_dfs[1].tail(size).copy()
""" 
        

def create_datasets(
        tickers:list[tuple[str, list[str]]],
        lock, 
        shared_tickers_cache:dict[str, int]) -> None:
    
    logging.info(f"Processing tickers: {tickers}")

    for ticker in tickers:
        ticker_symbvol:str = ticker[0]
        ticker_exchanges:list[str] = ticker[1]
        
        for exchange in ticker_exchanges:

            logging.info(f"Loading merged dataframes '{ticker_symbvol}' - '{exchange}' ...")

            dfs = load_merged_dataframes(ticker_symbvol, exchange, lock, shared_tickers_cache)

            enriched_dfs = {}

            for minute_multiplier, df in dfs.items():

                # if minute_multiplier != 390:
                #    continue

                logging.info(f"Processing '{ticker_symbvol}' - '{exchange}' - {minute_multiplier} ...")

                logging.info(f"Adding normalized time columns {minute_multiplier} ...")
                df = df_dt_utils.add_normalized_time_columns(df)

                if minute_multiplier < 390:
                    logging.info("Deleting non-trading hours records {minute_multiplier} ...")
                    df = df[(df['normalized_trading_time'] >= 0) & (df['normalized_trading_time'] < 1)]

                logging.info(f"Fixing trading price misprints {minute_multiplier} ...")
                df = fix_trading_price_misprints(df)

                logging.info(f"Processing empty values {minute_multiplier} ...")
                df = process_empty_values(df)

                logging.info(f"Reversing dataframe {minute_multiplier} ...")
                df = reverse_dataframe(df)
                
                logging.info(f"Adding technical indicators {minute_multiplier} ...")
                df, price_normalizing_columns, log_normalized_columns = df_tech_utils.add_technical_indicators(df, minute_multiplier)
                
                if minute_multiplier == 1:
                
                    logging.info(f"Adding price change labels {minute_multiplier} ...")
                    df = df_pa.addPriceChangeLabelsToDataFrame(df)                
                
                if minute_multiplier in [1, 3, 10, 30]:
                    logging.info(f"Adding volume profile {minute_multiplier} ...")
                    df = df_vp_utils.add_top_of_volume_profile(df)
                
                logging.info(f"Adding minute multiplier to column names {minute_multiplier} ...")
                df = add_minute_multiplier_to_column_names(df, minute_multiplier)

                enriched_dfs[minute_multiplier] = df

                logging.info(f"Saving dataset {minute_multiplier} ...")
                result_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
                df_ls.save_df(df, result_file_name)

                # create_test_multyrange_dataset(enriched_dfs, TEST_MULTYRANGE_DATASET_SIZE)

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
    logging.info("Waiting for all processes to finish ...")
    for process in processes:
        process.join()

    logging.info("All processes finished ...")
# ----------------------------

# test fix_trading_price_misprints(df:pd.DataFrame)
"""
df = pd.DataFrame({
    "TRADES_open": [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 1500.0, 16.0, 17.0, 18.0, 19.0, 20.0
    ]  
})

print(df)

df_fixed = fix_trading_price_misprints(df)

print(df_fixed)
"""

if __name__ == "__main__":
    
    ib_log.configure_logging("ib_raw_data_merger")

    logging.info(f"Starting {__file__} ...")

    fsu.create_required_folders()

    do_step()
