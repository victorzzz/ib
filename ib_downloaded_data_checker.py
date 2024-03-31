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
import df_loader_saver as df_ls

colmns_to_check = ["timestamp",
                   "MIDPOINT_open", "MIDPOINT_high", "MIDPOINT_low", "MIDPOINT_close",
                   "BID_open", "BID_high", "BID_low", "BID_close",
                   "ASK_open", "ASK_high", "ASK_low", "ASK_close",
                   "OPTION_IMPLIED_VOLATILITY_open", "OPTION_IMPLIED_VOLATILITY_high", "OPTION_IMPLIED_VOLATILITY_low", "OPTION_IMPLIED_VOLATILITY_close",
                   "TRADES_open", "TRADES_high", "TRADES_low", "TRADES_close", "TRADES_volume", "TRADES_average", "TRADES_barCount"]

def check_csv_files(raw_files:list[str]) -> bool:
    logging.info(f"Starting checking  ...")

    result:bool = True

    for file in raw_files:

        df:pd.DataFrame = df_ls.load_df(file)

        if df.empty:
            logging.error(f"File '{file}' is empty")
            result = False
            continue

        if len(df.columns) != len(colmns_to_check):
            logging.error(f"Number of columns in file '{file}' is not equal to {len(colmns_to_check)}")
            result = False
            continue

        # check presemts of required columns
        for col in colmns_to_check:
            if col not in df.columns:
                logging.error(f"Column '{col}' is missing in file '{file}'")
                result = False
                continue

    return result

def do_step() -> bool:
    raw_files:list[str] = list(fsu.iterate_files(cnts.data_folder))

    return check_csv_files(raw_files)

# ----------------------------

if __name__ == "__main__":
    
    ib_log.configure_logging("ib_downloaded_data_checker")

    logging.info(f"Starting {__file__} ...")

    fsu.create_required_folders()

    result:bool = do_step()
    return_code:int = 0 if result else 1

    # return return_code to OS
    exit(return_code)