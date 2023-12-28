import multiprocessing
import constants as cnts
import pandas as pd
from os.path import exists
import numpy as np
from typing import List
from typing import Dict
import ib_tickers as ib_tckrs
import file_systen_utils as fsu
import df_date_time_utils as df_dt_utils

def merge_csv_files(tickers:List[Dict[str, Dict[str, int]]], raw_files:List[str]):
    print(f"Starting rocessing '{', '.join(tickers)}' ...")

    for ticker in tickers:

        for ticker_info in ticker.items():
            ticker_symbvol:str = ticker_info[0]
            ticker_contracts:Dict[str,int] = ticker_info[1]
            
            for contarct in ticker_contracts.items():
                exchange:str = contarct[0]
                contract_id:int = contarct[1]

                for minute_multiplier in cnts.minute_multipliers:
                    filtered_raw_files:List[str] = [file for file in raw_files if file.startswith(f"{cnts.data_folder}/{ticker_symbvol}-{contract_id}--ib--{minute_multiplier:.0f}--minute--")]

                    if len(filtered_raw_files) == 0:
                        continue

                    merged_file_name = f"{cnts.merged_data_folder}/{ticker_symbvol}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged.csv"

                    merged_data_frame:pd.DataFrame = pd.DataFrame()
                    if exists(merged_file_name):
                        merged_data_frame = pd.read_csv(merged_file_name, index_col=0)

                    for raw_file in filtered_raw_files:
                        
                        raw_data_frame:pd.DataFrame = pd.read_csv(raw_file, index_col=0)
                        df_dt_utils.add_normalized_time_columns(raw_data_frame)

                        raw_data_frame = raw_data_frame.copy() # performance optimization
                        merged_data_frame = pd.concat([merged_data_frame, raw_data_frame], axis=0)
                    
                    merged_data_frame.sort_values(by='timestamp', inplace=True)
                    merged_data_frame.drop_duplicates(inplace=True)

                    print(f"Saving {merged_file_name} ...")
                    merged_data_frame.to_csv(merged_file_name, index=False)

                    print(f"Archiving {', '.join(filtered_raw_files)} ...")
                    for raw_file in filtered_raw_files:
                        fsu.move_file_to_folder(raw_file, cnts.data_archived_folder)

def do_step():
    processes = []

    raw_files:List[str] = list(fsu.iterate_files(cnts.data_folder))

    for tikers_batch in ib_tckrs.get_all_tickets_batches(cnts.complex_processing_batch_size):
        print("-------------------------------------")
        print(f"Group '{', '.join(tikers_batch)}' ...")
        print("-------------------------------------")

        process = multiprocessing.Process(target=merge_csv_files, args=(tikers_batch, raw_files,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    print(f"Waiting for '{', '.join(tikers_batch)}' ...")
    for process in processes:
        process.join()

# ----------------------------

if __name__ == "__main__":
    do_step()