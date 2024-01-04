from typing import Tuple
from typing import Dict
from typing import List
from typing import Optional

from os.path import exists
import time
import datetime as dt

from ib_insync import IB, Contract, BarData

import multiprocessing
import pandas as pd

import constants as cnts
import ib_constants as ib_cnts
import ib_tickers as ib_tckrs
import date_time_utils as dt_utils

#from_date:dt.date = dt.datetime.now().date()

from_date:dt.date = dt.datetime(2023, 12, 24)

fb_from_date:dt.date = dt.datetime(2022, 6, 8).date()
meta_to_day:dt.date = dt.datetime(2022, 6, 9).date()

ten_years_days:int = 366 * 10

minute_to_days_for_iteration = 10

def get_contract_for_contract_id(ib_client:IB, contract_id:int) -> Contract:

    qualifyContractStartTime = time.time()    
    contract = Contract(conId=contract_id)
    ib_client.qualifyContracts(contract)
    qualifyContractEndTime = time.time()
    print(f"qualifyContract delay {qualifyContractEndTime - qualifyContractStartTime}")

    return contract

def get_bars_for_contract(
        ib_client:IB,
        contract:Contract,
        data_type:str,
        duration_days:int,
        date_to:dt.date,
        minute_multiplier:float) -> Tuple[Optional[List[BarData]], float]:
    
        bars:Optional[List[BarData]] = None

        reqHistoricalDataStartTime = time.time()
        
        try:

            bars = ib_client.reqHistoricalData(
                contract = contract,
                endDateTime = date_to,
                durationStr = f"{duration_days} D",
                barSizeSetting = cnts.minute_multipliers[minute_multiplier],
                whatToShow=data_type,
                useRTH = True
            )

        except Exception as ex:
            print(f"Downloading error {ex} for {data_type} {contract.symbol}-{contract.conId} {contract.exchange} {minute_multiplier:.0f} minute(s). {date_to}")

        finally:
            reqHistoricalDataEndTime = time.time()
            reqHistoricalDataDelay = reqHistoricalDataEndTime - reqHistoricalDataStartTime
            print(f"reqHistoricalData {data_type} {contract.symbol}-{contract.conId} {contract.exchange} {minute_multiplier:.0f} minute(s).{date_to} delay {reqHistoricalDataDelay}")

        return (bars, reqHistoricalDataDelay)

def bars_to_dataframe(data_type:str, bars:List[BarData]) -> pd.DataFrame:

    bars_to_save:Optional[List[Dict[str, float]]] = None

    if (data_type == "TRADES"):
        bars_to_save = [
            {
                "timestamp": dt_utils.bar_date_to_epoch_time(bar.date),
                "TRADES_open": bar.open,
                "TRADES_high": bar.high,
                "TRADES_low": bar.low,
                "TRADES_close": bar.close,
                "TRADES_volume": bar.volume,
                "TRADES_average": bar.average,
                "TRADES_barCount": bar.barCount
            } 
            for bar in bars]
    else:
        bars_to_save = [
            {
                "timestamp": dt_utils.bar_date_to_epoch_time(bar.date),
                f"{data_type}_open": bar.open,
                f"{data_type}_high": bar.high,
                f"{data_type}_low": bar.low,
                f"{data_type}_close": bar.close
            } 
            for bar in bars]

    df = pd.DataFrame(bars_to_save)
    df.set_index('timestamp', inplace=True)

    return df

def download_stock_bars(
        date:dt.date, 
        ib_client:IB, 
        ticker_info:Tuple[str, Dict[str,int]], 
        minute_multiplier:float, 
        save_as:Optional[str] = None, 
        max_days_history:int = ten_years_days):
    
    iteration_time_delta_days = int(minute_to_days_for_iteration * minute_multiplier)
    iteration_time_delta = dt.timedelta(days = iteration_time_delta_days)

    maximum_time_delta = dt.timedelta(days = max_days_history) + dt.timedelta(days = minute_multiplier)
    limit_date = date - maximum_time_delta

    ticker:str = ticker_info[0]
    ticker_con_ids:Dict[str,int] = ticker_info[1]

    # read previously downloaded and merged data
    """
    merged_data_file_name = f"{cnts.merged_data_folder}/{ticker}--price-candle--{minute_multiplier}--minute.csv"

    if exists(merged_data_file_name):

        merged_df = pd.read_csv(merged_data_file_name)

        last_record = merged_df.tail(1)
        las_time_stamp = last_record['timestamp'].values[0]

        last_date = cnts.nyse_msec_timestamp_to_date_time(las_time_stamp).date()

        limit_date = max(limit_date, last_date)
    """

    for exchange, ticker_con_id in ticker_con_ids.items():

        if (exchange != "TSX"):
            continue

        print(f"Processing {ticker} {ticker_con_id} {minute_multiplier:.0f} minute(s). Date range: {date} .. {limit_date}")

        contract:Contract = get_contract_for_contract_id(ib_client, ticker_con_id)

        processing_date = date

        while processing_date > limit_date:
            date_to = processing_date
            date_to_str = date_to.strftime('%Y-%m-%d')

            tiker_to_save = ticker if (save_as is None) else save_as

            file_name = f"{cnts.data_folder}/{tiker_to_save}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}.csv"
            if exists(file_name):
                print(f"File {file_name} exists")
            else:

                final_data_frame:Optional[pd.DataFrame] = None

                last_dates:List[dt.date] = []

                for data_type in ib_cnts.hist_data_types:
                    
                    bars:Optional[List[BarData]]
                    reqHistoricalDataDelay:float
                    
                    bars, reqHistoricalDataDelay= get_bars_for_contract(
                        ib_client,
                        contract,
                        data_type,
                        iteration_time_delta_days,
                        date_to,
                        minute_multiplier)

                    if (bars is None or len(bars) == 0):
                        print(f"!!!! Empty data for {data_type} {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                        continue

                    df:pd.DataFrame = bars_to_dataframe(data_type, bars)

                    if (final_data_frame is None):
                        final_data_frame = df
                    else:
                        final_data_frame = pd.concat([final_data_frame, df], axis=1, sort=True)

                    waitTime = max(0.1, 10.0 - reqHistoricalDataDelay + 0.1)
                    print(f"waiting for {waitTime} seconds")
                    time.sleep(waitTime)

                if (final_data_frame is None):
                    print(f"***** Empty data for {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                else:
                    print(f"Saving file {file_name}. {len(final_data_frame)}") 
                    final_data_frame.to_csv(file_name)

            processing_date = processing_date - iteration_time_delta - dt.timedelta(days=1)

def download_stock_bars_for_tickers(
        tickers:List[Tuple[str, Dict[str, int]]],
        port:int,
        client_id:int,
        host:str) :
    
    print(f"download_stock_bars_for_tickers port:{port} client_id:{client_id} host:{host} -- tickers:{tickers}")

    ib_client:IB = IB()
    ib_client.connect(
        readonly=True,
        port=port,
        clientId=client_id,
        host=host)

    for ticker_info in tickers:
        if (ticker_info[0] == "META"):
            continue

        for multiplier in cnts.minute_multipliers:
            download_stock_bars(from_date, ib_client, ticker_info, multiplier)


def do_step():

    all_tickers = ib_tckrs.get_all_tickers_list()
    even_tickers = ib_tckrs.get_even_items(all_tickers)
    odd_tickers = ib_tckrs.get_odd_items(all_tickers)

    process1 = multiprocessing.Process(
        target=download_stock_bars_for_tickers,
        args=(even_tickers,
            ib_cnts.hist_data_loader_live_port,
            ib_cnts.hist_data_loader_live_client_id,
            ib_cnts.hist_data_loader_live_host))

    process1.start()

    process2 = multiprocessing.Process(
        target=download_stock_bars_for_tickers,
        args=(odd_tickers,
            ib_cnts.hist_data_loader_paper_port,
            ib_cnts.hist_data_loader_paper_client_id,
            ib_cnts.hist_data_loader_paper_host))

    process2.start()    

    process1.join()
    process2.join()

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
