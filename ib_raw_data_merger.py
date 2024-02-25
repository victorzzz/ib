import multiprocessing
import constants as cnts
import pandas as pd
from os.path import exists
import numpy as np
from typing import Optional
import ib_tickers as ib_tckrs
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import logging
import ib_logging as ib_log
import ib_tickers_cache as ib_tickers_cache

def merge_csv_files(
        tickers:list[tuple[str, list[str]]],
        raw_files:list[str],
        lock, 
        shared_tickers_cache:dict[str, int]):
    
    ib_log.configure_logging("ib_raw_data_merger")

    processed_ticker_symbols = [ticker[0] for ticker in tickers]
    logging.info(f"Starting rocessing '{', '.join(processed_ticker_symbols)}' ...")

    for ticker in tickers:

        ticker_symbvol:str = ticker[0]
        ticker_exchanges:list[str] = ticker[1]
        
        for exchange in ticker_exchanges:

            contract_id:Optional[int] = ib_tickers_cache.get_contact_id(ticker_symbvol, exchange, lock, shared_tickers_cache)

            if contract_id is None:
                logging.warning(f"Contract ID not found for '{ticker_symbvol}' '{exchange}' ...")
                continue

            for minute_multiplier in cnts.minute_multipliers:
                filtered_raw_files:list[str] = [file for file in raw_files 
                                if file.startswith(f"{cnts.data_folder}\\{ticker_symbvol}-{contract_id}--ib--{minute_multiplier:.0f}--minute--")]

                if len(filtered_raw_files) == 0:
                    logging.warning(f"No files for '{ticker_symbvol}-{contract_id}--ib--{minute_multiplier:.0f}--minute--' ...")
                    continue

                merged_file_name = f"{cnts.merged_data_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged.csv"

                merged_data_frame:pd.DataFrame = pd.DataFrame()
                if exists(merged_file_name):
                    merged_data_frame = pd.read_csv(merged_file_name)

                for raw_file in filtered_raw_files:
                    
                    raw_data_frame:pd.DataFrame = pd.read_csv(raw_file)
                    raw_data_frame.sort_values(by='timestamp', inplace=True, ascending=False)

                    merged_data_frame = pd.concat([raw_data_frame, merged_data_frame], axis=0)
                    
                    duplicated_array = merged_data_frame.duplicated(subset=['timestamp'])

                    if(duplicated_array.sum() > 0):
                        logging.error(f"MERGER !!!! Found {duplicated_array.sum()} duplicates in {raw_file} ...")

                        duplicates_data_frame:pd.DataFrame = pd.DataFrame(duplicated_array)
                        duplicates_file_name:str = f"{cnts.merged_data_duplicates_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged-DUP.csv"
                        duplicates_data_frame.to_csv(duplicates_file_name, index=False)

                merged_data_frame.sort_values(by='timestamp', inplace=True, ascending=False)
                merged_data_frame.drop_duplicates(subset=['timestamp'], inplace=True)

                logging.info(f"Saving {merged_file_name} ...")
                merged_data_frame.to_csv(merged_file_name, index=False)

                logging.info(f"Archiving {', '.join(filtered_raw_files)} ...")
                for raw_file in filtered_raw_files:
                    fsu.move_file_to_folder(raw_file, cnts.data_archived_folder)


def do_step():
    processes = []

    raw_files:list[str] = list(fsu.iterate_files(cnts.data_folder))

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    for tikers_batch in ib_tckrs.get_selected_tickers_batches(cnts.complex_processing_batch_size):
        processed_ticker_symbols = [ticker[0] for ticker in tikers_batch]
               
        logging.info("-------------------------------------")
        logging.info(f"Group '{', '.join(processed_ticker_symbols)}' ...")
        logging.info("-------------------------------------")

        process = multiprocessing.Process(
            target=merge_csv_files, 
            args=(tikers_batch, raw_files, lock, shared_dict))
        
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    logging.info(f"Waiting for '{', '.join(processed_ticker_symbols)}' ...")
    for process in processes:
        process.join()

# ----------------------------

if __name__ == "__main__":
    
    ib_log.configure_logging("ib_raw_data_merger")

    logging.info(f"Starting {__file__} ...")

    fsu.create_required_folders()

    do_step()