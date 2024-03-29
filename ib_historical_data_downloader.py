
from typing import Optional
import math

from os.path import exists
import time
import datetime as dt

from ib_insync import IB, Contract, BarData, ContractDetails

import multiprocessing
import pandas as pd

import constants as cnts
import ib_constants as ib_cnts
import ib_tickers as ib_tckrs
import date_time_utils as dt_utils
import file_system_utils as fs_utils
import logging
import ib_logging as ib_log
import ib_tickers_cache as ib_tickers_cache

import df_loader_saver as df_ls

#from_date:dt.date = dt.datetime.now().date()

from_date:dt.date = dt.date(2024, 2, 16)

fb_from_date:dt.date = dt.date(2022, 6, 8)
meta_to_day:dt.date = dt.date(2022, 6, 9)

# ten_years_days:int = 366 * 10
# fifteen_years_days:int = 366 * 15

twenty_years_days:int = 366 * 20
minute_to_days_for_iteration = 10

max_request_attempts:int = 20
failed_request_delay:float = 10.0

def get_contract_for_symbol_and_exchange(ib_client:IB, symbol:str, exchange:str, lock, shared_tickers_cache:dict[str, int]) -> Contract:

    contract_id:Optional[int] = ib_tickers_cache.get_contact_id(symbol, exchange, lock, shared_tickers_cache)

    if (contract_id is None):
        reqContractDetailsStartTime = time.time()    
        contract = Contract(secType = "STK", symbol=symbol, exchange=exchange)
        contract_details_list:list[ContractDetails] = ib_client.reqContractDetails(contract)
        reqContractDetailsEndTime = time.time()

        logging.info(f"reqContractDetails latency {reqContractDetailsEndTime - reqContractDetailsStartTime}")

        ib_client.sleep(3)

        if (len(contract_details_list) == 0):
            raise Exception(f"Contract details not found for {symbol} {exchange}")
        
        if (len(contract_details_list) > 1):
            raise Exception(f"Multiple contract details found for {symbol} {exchange}")

        contract = contract_details_list[0].contract

        if (contract is None):
            raise Exception(f"Contract is None {symbol} {exchange}")

        contract_id = contract.conId

        logging.info(f"Adding contract id to cache for {symbol} {exchange} - {contract_id}")
        ib_tickers_cache.add_contact_id(symbol, exchange, contract_id, lock, shared_tickers_cache)

    contract_to_qualify:Contract = Contract(conId=contract_id)
    ib_client.qualifyContracts(contract_to_qualify)
    logging.info(f"Contract qualified {contract_to_qualify.symbol}-{contract_to_qualify.conId} {contract_to_qualify.exchange}")
    logging.info(f"Qualified contract {contract_to_qualify}")

    ib_client.sleep(3)

    return contract_to_qualify

def get_bars_for_contract(
        ib_client:IB,
        contract:Contract,
        data_type:str,
        duration_days:int,
        date_to:dt.date,
        minute_multiplier:float) -> tuple[Optional[list[BarData]], float]:
    
        attempt:int = 1

        while True:
            bars:Optional[list[BarData]] = None

            reqHistoricalDataStartTime = time.time()
            
            try:
                
                furation_param = f"{duration_days} D"
                if duration_days > 30:
                        if duration_days < 360:
                            furation_param = f"{math.ceil(duration_days/30.0)} M"
                        else:
                            furation_param = f"{math.ceil(duration_days/360.0)} Y"

                bars = ib_client.reqHistoricalData(
                    contract = contract,
                    endDateTime = date_to,
                    durationStr = furation_param,
                    barSizeSetting = cnts.minute_multipliers[minute_multiplier],
                    whatToShow=data_type,
                    useRTH = True
                )

            except Exception as ex:
                logging.critical(f"Downloading error {ex} for {data_type} {contract.symbol}-{contract.conId} {contract.exchange} {minute_multiplier:.0f} minute(s). {date_to}")

            finally:
                reqHistoricalDataEndTime = time.time()
                reqHistoricalDataDelay = reqHistoricalDataEndTime - reqHistoricalDataStartTime
                logging.info(f"reqHistoricalData {data_type} {contract.symbol}-{contract.conId} {contract.exchange} {minute_multiplier:.0f} minute(s).{date_to} delay {reqHistoricalDataDelay}")

            if (bars is not None):
                return (bars, reqHistoricalDataDelay)
            
            attempt += 1

            if (attempt > max_request_attempts):
                logging.critical(f"Downloading max attemps {attempt}")
                return (bars, reqHistoricalDataDelay)

            logging.warning(f"DOWNLOADER !!!! NO data for {data_type} {contract.symbol}-{contract.conId} {contract.exchange} {minute_multiplier:.0f} minute(s). {date_to}. RETRYING attempt {attempt}")
            logging.debug(f"waiting for {failed_request_delay} seconds before retrying")
            ib_client.sleep(failed_request_delay)

def bars_to_dataframe(data_type:str, bars:list[BarData]) -> pd.DataFrame:

    bars_to_save:Optional[list[dict[str, float]]] = None

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

    return df

def get_oldest_date_from_saved_data(file_name:str) -> Optional[dt.date]:

    if not exists(file_name):
        return None

    df = pd.read_csv(file_name)

    if (len(df) == 0):
        return None

    epoch_time = df['timestamp'].min()
    if epoch_time is None:
        return None

    oldest_date = dt_utils.nyse_timestamp_to_date_time(int(epoch_time)).date()

    return oldest_date

def concat_dataframes(df1:pd.DataFrame, df2:pd.DataFrame, logContext:str) -> Optional[pd.DataFrame]:

    if (df1 is None):
        logging.warning(f"Concat {logContext} !! df1 is None")
        return df2

    if (df2 is None):
        logging.warning(f"Concat {logContext} !! df2 is None")
        return df1
    
    result_df = pd.merge_ordered(df1, df2, on='timestamp', how='outer')

    nan_before = result_df.isna().any()

    if nan_before.any():
        logging.warning(f"Concat {logContext} !! NaN found after merging")

    return result_df

nearest_data_head_cache:dict[str, dt.datetime] = {}

def get_nearest_data_head(ib_client:IB, contract:Contract, data_types_to_download:tuple[str, ...]) -> dt.datetime:
    key = f"{contract.conId}"
    if key in nearest_data_head_cache:
        return nearest_data_head_cache[key]

    headTimeStamps:list[dt.datetime] = []

    for data_type in data_types_to_download:
        headTimeStamp:dt.datetime = ib_client.reqHeadTimeStamp(contract = contract, whatToShow=data_type, useRTH = True)
        ib_client.sleep(3)
        headTimeStamps.append(headTimeStamp)

    maxHeadTimeStamp:dt.datetime = max(headTimeStamps)
    nearest_data_head_cache[key] = maxHeadTimeStamp

    return maxHeadTimeStamp

def get_last_merged_datetime(ticker:str, contract_id:int, exchange:str, minute_multiplier:float) -> Optional[dt.datetime]:
    file_name = f"{cnts.merged_data_folder}/{ticker}-{contract_id}-{exchange}--ib--{minute_multiplier:.0f}--minute--merged.csv"
    if not exists(file_name):
        return None
    
    df = csvt.get_dataframe_first_row_only(file_name)
    if (len(df) == 0):
        return None
    
    timestamp = df['timestamp'][0]

    return dt_utils.nyse_timestamp_to_date_time(int(timestamp))

def download_stock_bars(
        date:dt.date, 
        ib_client:IB, 
        ticker_info:tuple[str, list[str]], 
        minute_multiplier:float,
        data_types_to_download:tuple[str, ...],
        lock, 
        shared_tickers_cache:dict[str, int],
        save_as:Optional[str] = None, 
        max_days_history:int = twenty_years_days):
    
    iteration_time_delta_days = int(minute_to_days_for_iteration * minute_multiplier)
    iteration_time_delta = dt.timedelta(days = iteration_time_delta_days)

    maximum_time_delta = dt.timedelta(days = max_days_history) + dt.timedelta(days = minute_multiplier)
    limit_date = date - maximum_time_delta

    ticker:str = ticker_info[0]
    ticker_exchanges:list[str] = ticker_info[1]

    for exchange in ticker_exchanges:

        logging.info(f"Processing {ticker} {exchange} {minute_multiplier:.0f} minute(s). Date range: {date} .. {limit_date}")

        contract:Contract = get_contract_for_symbol_and_exchange(ib_client, ticker, exchange, lock, shared_tickers_cache)
        ticker_con_id = contract.conId

        processing_date = date

        last_merged_datetime = get_last_merged_datetime(ticker, ticker_con_id, exchange, minute_multiplier)
        if (last_merged_datetime is None):
            logging.info(f"No merged data for {ticker} {exchange} {minute_multiplier:.0f} minute(s)")
        else:
            logging.info(f"Last merged datetime found for {ticker} {exchange} {minute_multiplier:.0f} minute(s): {last_merged_datetime}")

            if (processing_date <= last_merged_datetime.date()):
                logging.info(f"Skipping {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute: '{processing_date}' <= '{last_merged_datetime}'")
                continue

            limit_date = max(limit_date, last_merged_datetime.date())

        nearest_data_head = get_nearest_data_head(ib_client, contract, data_types_to_download)
        logging.info(f"IBRK data head for {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute: {nearest_data_head}")
        
        nearest_data_head = nearest_data_head.date() + dt.timedelta(days=1)
        logging.info(f"Nearest data head for {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute: {nearest_data_head}")

        limit_date_for_contract = max(limit_date, nearest_data_head)

        logging.info(f"Limit date for {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute: {limit_date_for_contract}")

        while processing_date > limit_date_for_contract:
            date_to = processing_date
            date_to_str = date_to.strftime('%Y-%m-%d')

            tiker_to_save = ticker if (save_as is None) else save_as

            file_name = f"{cnts.data_folder}/{tiker_to_save}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}"
            if df_ls.is_df_exists(file_name):
                logging.info(f"File {file_name} exists")
                oldest_date_in_data_from_file = get_oldest_date_from_saved_data(file_name)
                if (oldest_date_in_data_from_file is None):
                    logging.warning(f"No data in existing file {file_name}")                
                    processing_date = processing_date - iteration_time_delta - dt.timedelta(days=1)
                else:
                    processing_date = oldest_date_in_data_from_file - dt.timedelta(days=1)
            else:

                final_data_frame:Optional[pd.DataFrame] = None

                oldest_dates:list[dt.date] = []

                for data_type in data_types_to_download:

                    bars:Optional[list[BarData]]
                    reqHistoricalDataDelay:float
                    
                    bars, reqHistoricalDataDelay= get_bars_for_contract(
                        ib_client,
                        contract,
                        data_type,
                        iteration_time_delta_days,
                        date_to,
                        minute_multiplier)

                    if (bars is None or len(bars) == 0):
                        logging.error(f"DOWNLOADER !!!! Empty data for {data_type} {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                        raise Exception(f"DOWNLOADER !!!! Empty data for {data_type} {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")

                    oldest_date = dt_utils.bar_date_to_date(bars[0].date)
                    oldest_dates.append(oldest_date)
                    
                    df:pd.DataFrame = bars_to_dataframe(data_type, bars)

                    if (final_data_frame is None):
                        final_data_frame = df
                    else:
                        concatenated_data_frame =  concat_dataframes(
                            final_data_frame, df, 
                            f"{data_type} {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                        if (concatenated_data_frame is None):
                            logging.error(f"DOWNLOADER !!!! Dataframe is non after merging {data_type} {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                            
                            investigation_file_name = f"{cnts.error_investigation_folder}/{data_type}--{tiker_to_save}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}"
                            df_ls.save_df(df, investigation_file_name, "csv")

                            # df.to_csv(investigation_file_name, index=False)
                        else:
                            final_data_frame = concatenated_data_frame

                    waitTime = max(0.1, 10.0 - reqHistoricalDataDelay)
                    logging.debug(f"waiting for {waitTime} seconds")
                    ib_client.sleep(waitTime)

                if (final_data_frame is None):
                    logging.warning(f"***** Empty data for {ticker}-{ticker_con_id}--ib--{minute_multiplier:.0f}--minute--{iteration_time_delta_days}--{date_to_str}")
                else:
                    logging.info(f"Saving file {file_name}. {len(final_data_frame)}")

                    df_ls.save_df(final_data_frame, file_name)
                    # final_data_frame.to_csv(file_name, index=False)

                if (len(oldest_dates) > 0):
                    min_oldest_date = min(oldest_dates)
                    processing_date = min_oldest_date - dt.timedelta(days=1)
                else:
                    processing_date = processing_date - iteration_time_delta - dt.timedelta(days=1)

def download_stock_bars_for_tickers(
        tickers:list[tuple[str, list[str]]],
        port:int,
        client_id:int,
        host:str,
        lock,
        shared_tickers_cache:dict[str, int]) :
    
    ib_log.configure_logging("ib_historical_data_downloader")

    logging.info(f"download_stock_bars_for_tickers port:{port} client_id:{client_id} host:{host} -- tickers:{tickers}")

    try:

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
                download_stock_bars(
                    from_date,
                    ib_client,
                    ticker_info,
                    multiplier,
                    ib_cnts.hist_data_types_reduced,
                    lock,
                    shared_tickers_cache)

    except Exception as ex:
        logging.exception(f"download_stock_bars_for_tickers port:{port} client_id:{client_id} host:{host} -- tickers:{tickers} {ex}")
    finally:
        logging.info(f"download_stock_bars_for_tickers port:{port} client_id:{client_id} host:{host} -- tickers:{tickers} disconnecting")
        ib_client.disconnect()

def do_step():

    selected_tickers:list[tuple[str, list[str]]] = ib_tckrs.get_selected_tickers_list()
    even_tickers:list[tuple[str, list[str]]] = ib_tckrs.get_even_items(selected_tickers)
    odd_tickers:list[tuple[str, list[str]]] = ib_tckrs.get_odd_items(selected_tickers)

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    process1 = multiprocessing.Process(
        target=download_stock_bars_for_tickers,
        args=(even_tickers,
            ib_cnts.hist_data_loader_live_port,
            ib_cnts.hist_data_loader_live_client_id,
            ib_cnts.hist_data_loader_live_host,
            lock,
            shared_dict))

    process1.start()

    process2 = multiprocessing.Process(
        target=download_stock_bars_for_tickers,
        args=(odd_tickers,
            ib_cnts.hist_data_loader_paper_port,
            ib_cnts.hist_data_loader_paper_client_id,
            ib_cnts.hist_data_loader_paper_host,
            lock,
            shared_dict))

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
    
    ib_log.configure_logging("ib_historical_data_downloader")

    logging.info(f"Starting {__file__} ...")

    fs_utils.create_required_folders()

    do_step()
