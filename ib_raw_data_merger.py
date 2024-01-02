import multiprocessing
import constants as cnts
import pandas as pd
from os.path import exists
import numpy as np
from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional
import ib_tickers as ib_tckrs
import file_systen_utils as fsu
import df_date_time_utils as df_dt_utils

def merge_csv_files(tickers:List[Tuple[str, Dict[str, int]]], raw_files:List[str]):
    processed_ticker_symbols = [ticker[0] for ticker in tickers]
    print(f"Starting rocessing '{', '.join(processed_ticker_symbols)}' ...")

    for ticker in tickers:

        ticker_symbvol:str = ticker[0]
        ticker_contracts:Dict[str,int] = ticker[1]
        
        for contarct in ticker_contracts.items():
            exchange:str = contarct[0]
            contract_id:int = contarct[1]

            for minute_multiplier in cnts.minute_multipliers:
                filtered_raw_files:List[str] = [file for file in raw_files if file.startswith(f"{cnts.data_folder}\\{ticker_symbvol}-{contract_id}--ib--{minute_multiplier:.0f}--minute--")]

                if len(filtered_raw_files) == 0:
                    continue

                merged_file_name = f"{cnts.merged_data_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged.csv"

                merged_data_frame:pd.DataFrame = pd.DataFrame()
                if exists(merged_file_name):
                    merged_data_frame = pd.read_csv(merged_file_name)

                    merged_data_frame['timestamp_for_index'] = merged_data_frame['timestamp']
                    merged_data_frame = merged_data_frame.set_index('timestamp_for_index')

                for raw_file in filtered_raw_files:
                    
                    raw_data_frame:pd.DataFrame = pd.read_csv(raw_file)

                    raw_data_frame['timestamp_for_index'] = raw_data_frame['timestamp']
                    raw_data_frame = raw_data_frame.set_index('timestamp_for_index')

                    df_dt_utils.add_normalized_time_columns(raw_data_frame)

                    raw_data_frame = raw_data_frame.copy() # performance optimization

                    merged_data_frame = pd.concat([merged_data_frame, raw_data_frame], axis=0)
                    
                    duplicated_array = merged_data_frame.duplicated(subset=['timestamp'])
                    duplicates_data_frame:pd.DataFrame = pd.DataFrame(duplicated_array)

                    if(duplicated_array.sum() > 0):

                        print("raw_data_frame: ")
                        print(raw_data_frame)
                        
                        print("merged_data_frame: ")
                        print(merged_data_frame)

                        print("duplicates_data_frame: ")
                        print(duplicates_data_frame)
                
                merged_data_frame.sort_values(by='timestamp', inplace=True)

                merged_data_frame_before_dropping_duplicates = merged_data_frame.copy()

                merged_data_frame.drop_duplicates(subset=['timestamp'], inplace=True)

                print(f"Saving {merged_file_name} ...")
                merged_data_frame.to_csv(merged_file_name, index=False)

                duplicates_file_name:str = f"{cnts.merged_data_duplicates_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged-DUP.csv"

                print(f"Saving {merged_file_name}-with-duplicates.csv ...")
                merged_data_frame_before_dropping_duplicates.to_csv(f"{merged_file_name}-with-duplicates.csv", index=False)

                print(f"Saving {duplicates_file_name} ...")
                merged_data_frame_before_dropping_duplicates.to_csv(duplicates_file_name, index=False)

                print(f"Archiving {', '.join(filtered_raw_files)} ...")
                # for raw_file in filtered_raw_files:
                    # fsu.move_file_to_folder(raw_file, cnts.data_archived_folder)

def do_step():
    processes = []

    raw_files:List[str] = list(fsu.iterate_files(cnts.data_folder))

    for tikers_batch in ib_tckrs.get_all_tickets_batches(cnts.complex_processing_batch_size):
        processed_ticker_symbols = [ticker[0] for ticker in tikers_batch]
               
        print("-------------------------------------")
        print(f"Group '{', '.join(processed_ticker_symbols)}' ...")
        print("-------------------------------------")

        process = multiprocessing.Process(target=merge_csv_files, args=(tikers_batch, raw_files,))
        processes.append(process)
        process.start()

        # Wait for all processes to finish
        print(f"Waiting for '{', '.join(processed_ticker_symbols)}' ...")
        for process in processes:
            process.join()

# ----------------------------

if __name__ == "__main__":
    do_step()