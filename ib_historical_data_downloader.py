from typing import Tuple

from os.path import exists
import time
import datetime as dt

from ib_insync import *

import pandas as pd

import constants as cnts
import ib_constants as ib_cnts
import ib_tickers as ib_tckrs
import SECRETS as secrets

from_date:dt.date = dt.datetime.now().date()
fb_from_date:dt.date = dt.datetime(2022, 6, 8).date()
meta_to_day:dt.date = dt.datetime(2022, 6, 9).date()

ten_years_days:int = 365 * 10

# ticker: (Symbol, IBKR_ConID)
def download_stock_bars(
        date:dt.date, 
        ib_client:IB, 
        ticker_info:Tuple[str, int], 
        minute_multiplier:float, 
        save_as:str = None, 
        max_days_history:int = ten_years_days):
    
    iteration_time_delta = dt.timedelta(days=4 * minute_multiplier)

    maximum_time_delta = dt.timedelta(days=max_days_history)
    limit_date = date - maximum_time_delta

    ticker = ticker_info[0]
    ticker_con_ids = ticker_info[1]

    # read previously downloaded and merged data
    merged_data_file_name = f"{cnts.merged_data_folder}/{ticker}--price-candle--{minute_multiplier}--minute.csv"

    if exists(merged_data_file_name):

        merged_df = pd.read_csv(merged_data_file_name)

        last_record = merged_df.tail(1)
        las_time_stamp = last_record['timestamp'].values[0]

        last_date = cnts.nyse_msec_timestamp_to_date_time(las_time_stamp).date()

        limit_date = max(limit_date, last_date)

    while date > limit_date:
        date_from = date - iteration_time_delta
        if (date_from < limit_date):
            date_from = limit_date

        date_from_str = date_from.strftime('%Y-%m-%d')

        date_to = date
        date_to_str = date_to.strftime('%Y-%m-%d')

        tiker_to_save = ticker if (save_as is None) else save_as

        duration_days = (date_to - date_from).days
        if (duration_days == 0):
            duration_days = 1

        for ticker_con_id_index in range(len(ticker_con_ids)):
            ticker_con_id = ticker_con_ids[ticker_con_id_index]

            file_name = f"{cnts.data_folder}/{tiker_to_save}-{ticker_con_id_index}--ib--{minute_multiplier}--minute--{date_from_str}--{date_to_str}.csv"
            if exists(file_name):
                print(f"File {file_name} exists")
            else:

                final_data_frame:pd.DataFrame = pd.DataFrame()

                for data_type in ib_cnts.hist_data_types:

                    print(f"Call {data_type} {ticker}-{ticker_con_id_index}--ib--{minute_multiplier}--minute--{date_from_str}--{date_to_str}")

                    try:
                        contract = Contract(conId=ticker_con_id)

                        ib_client.qualifyContracts(contract)

                        bars = ib_client.reqHistoricalData(
                            contract = contract,
                            endDateTime = date_to,
                            durationStr = f"{duration_days} D",
                            barSizeSetting = cnts.minute_multipliers[minute_multiplier],
                            whatToShow=data_type,
                            useRTH = True
                        )

                        df = pd.DataFrame(bars)
                        pd.concat([final_data_frame, df], axis=1)

                    except Exception as ex:
                        print(f"Downloading error {ex}")

                    print(" ")
                    time.sleep(0.5)

                df.to_csv(file_name, index=False)
                print(f"Saving file {file_name}. {len(bars)}") 

            date = date - iteration_time_delta - dt.timedelta(days=1)

def do_step():
    ib_client:IB = IB()
    ib_client.connect(
        readonly=True,
        port=ib_cnts.hist_data_loader_port,
        clientId=ib_cnts.hist_data_loader_client_id,
        host=ib_cnts.hist_data_loader_host)

    # all except META
    for ticker_info in ib_tckrs.get_all_tickers_list():
        if (ticker_info[0] == "META"):
            continue

        for multiplier in cnts.minute_multipliers:
            download_stock_bars(from_date, ib_client, ticker_info, multiplier)

    # META / FB

    """
    for ticker_info[0] in ["FB"]:
        for multiplier in cnts.minute_multipliers:
            download_stock_bars(fb_from_date, ib_client, ticker_info, multiplier, save_as="META")

    for ticker_info[0] in ["META"]:
        for multiplier in cnts.minute_multipliers:
            download_stock_bars(from_date, ib_client, ticker_info, multiplier, meta_to_day)
    """
            
if __name__ == "__main__":
    do_step()
