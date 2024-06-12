import typing as tp

import logging

import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel

from darts.metrics import mape
from darts.utils.likelihood_models import QuantileRegression

import constants as cnts
import df_loader_saver as df_ls
import df_date_time_utils as df_dt_utils

TARGET_COLUMNS:list[str] = [
    '1m_MIDPOINT_close', '1m_BID_close', '1m_ASK_close',
    ]

COVARIATE_COLUMNS:list[str] = [
    '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low',
    '1m_BID_open', '1m_BID_high', '1m_BID_low',
    '1m_ASK_open', '1m_ASK_high', '1m_ASK_low',
    '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
    '1m_TRADES_volume', '1m_TRADES_average',
    '1m_normalized_day_of_week', '1m_normalized_week', '1m_normalized_trading_time',
    '1m__t_MFI_TRADES_average_7', '1m__t_MFI_TRADES_average_14', '1m__t_MFI_TRADES_average_21',
    '1m__t_RSI_TRADES_average_7', '1m__t_RSI_TRADES_average_14', '1m__t_RSI_TRADES_average_21',
    '1m__t_CCI_TRADES_average_7', '1m__t_CCI_TRADES_average_14', '1m__t_CCI_TRADES_average_21',
    '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3', 
    '1m__t_STOCH_k_TRADES_average_21_4', '1m__t_STOCH_d_TRADES_average_21_4',
    ]

PRICE_SCALING_COLUMNS:list[str] = [
    '1m_MIDPOINT_close', '1m_BID_close', '1m_ASK_close',
     '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low',
    '1m_BID_open', '1m_BID_high', '1m_BID_low',
    '1m_ASK_open', '1m_ASK_high', '1m_ASK_low',
    '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
    '1m_TRADES_average',
    ]

COVARIATE_PRICE_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x in PRICE_SCALING_COLUMNS]

PRICE_FITTING_COLUMN:str = '1m_MIDPOINT_close'

VOLUME_SCALING_COLUMNS:list[str] = ['1m_TRADES_volume']
VOLUME_FITTING_COLUMN:str = '1m_TRADES_volume'

COVARIATE_VOLUME_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x in VOLUME_SCALING_COLUMNS]

COVARIATE_NOT_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x not in PRICE_SCALING_COLUMNS and x not in VOLUME_SCALING_COLUMNS]

# returns target(trane, val, test), covariate(train, val, test), price_scaler, volume_scaler
def prepare_traine_val_test_datasets(symbol:str, exchange:str, tail:float | None = None) -> tuple[tuple[TimeSeries, TimeSeries, TimeSeries], tuple[TimeSeries, TimeSeries, TimeSeries], Scaler, Scaler]:
    target_ts, covariate_ts = prepare_timeseries_for_symbol(symbol, exchange, tail)

    price_scaler = Scaler(name='price_scaler')
    volume_scaler = Scaler(name='volume_scaler')
    
    # Split the time series
    train_target, val_test_target = target_ts.split_after(0.9)
    val_target, test_target = val_test_target.split_after(0.5)
    
    train_covariate, val_test_covariate = covariate_ts.split_after(0.9)
    val_covariate, test_covariate = val_test_covariate.split_after(0.5)
    
    price_scaler.fit(train_target[PRICE_FITTING_COLUMN])
    volume_scaler.fit(train_covariate[VOLUME_FITTING_COLUMN])    
    
    train_target = price_scaler.transform(train_target)
    val_target = price_scaler.transform(val_target)
    test_target = price_scaler.transform(test_target)
    
    traine_cv_volume = volume_scaler.transform(train_covariate[COVARIATE_VOLUME_SCALING_COLUMNS])
    traine_cv_price = price_scaler.transform(train_covariate[COVARIATE_PRICE_SCALING_COLUMNS])
    traine_cv_not_scaling = train_covariate[COVARIATE_NOT_SCALING_COLUMNS]
    
    val_cv_volume = volume_scaler.transform(val_covariate[COVARIATE_VOLUME_SCALING_COLUMNS])
    val_cv_price = price_scaler.transform(val_covariate[COVARIATE_PRICE_SCALING_COLUMNS])
    val_cv_not_scaling = val_covariate[COVARIATE_NOT_SCALING_COLUMNS]
    
    test_cv_volume = volume_scaler.transform(test_covariate[COVARIATE_VOLUME_SCALING_COLUMNS])
    test_cv_price = price_scaler.transform(test_covariate[COVARIATE_PRICE_SCALING_COLUMNS])
    test_cv_not_scaling = test_covariate[COVARIATE_NOT_SCALING_COLUMNS]
    
    if isinstance(train_target, TimeSeries) and isinstance(val_target, TimeSeries) and isinstance(test_target, TimeSeries) \
        and isinstance(train_covariate, TimeSeries) and isinstance(val_covariate, TimeSeries) and isinstance(test_covariate, TimeSeries) \
        and isinstance(traine_cv_volume, TimeSeries) and isinstance(traine_cv_price, TimeSeries) and isinstance(traine_cv_not_scaling, TimeSeries) \
        and isinstance(val_cv_volume, TimeSeries) and isinstance(val_cv_price, TimeSeries) and isinstance(val_cv_not_scaling, TimeSeries) \
        and isinstance(test_cv_volume, TimeSeries) and isinstance(test_cv_price, TimeSeries) and isinstance(test_cv_not_scaling, TimeSeries) :
    
        train_covariate = concatenate([traine_cv_volume, traine_cv_price, traine_cv_not_scaling])
        val_covariate = concatenate([val_cv_volume, val_cv_price, val_cv_not_scaling])
        test_covariate = concatenate([test_cv_volume, test_cv_price, test_cv_not_scaling])
    
        return (train_target, val_target, test_target), (train_covariate, val_covariate, test_covariate), price_scaler, volume_scaler
    else:
        raise ValueError("One of the time series is not a TimeSeries object.")


# returns (target_timeseries, covariate_timeseries)
def prepare_timeseries_from_dataframe(df:pd.DataFrame) -> tuple[TimeSeries, TimeSeries]:
    
    target_ts = TimeSeries.from_dataframe(
        df, value_cols = TARGET_COLUMNS)
    
    covariate_ts = TimeSeries.from_dataframe(
        df, value_cols = COVARIATE_COLUMNS)  
    return (target_ts, covariate_ts)

def prepare_timeseries_for_symbol(symbol:str, exchange:str, tail:float | None = None) -> tuple[TimeSeries, TimeSeries]:
    df_1m:pd.DataFrame = load_prepared_raw_datasets(symbol, exchange)[0]
    
    if tail is not None:
        df_1m = df_1m.tail(int(len(df_1m) * tail))
    
    df_1m['1m_timestamp'] = df_1m['1m_timestamp'].astype('int64')
    # df_1m['1m_datetime'] = df_dt_utils.timestamp_to_datetime(df_1m['1m_timestamp'])
    df_1m.set_index('1m_timestamp', inplace=True)
    
    return prepare_timeseries_from_dataframe(df_1m)

def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str) -> list[pd.DataFrame]:
    logging.info(f"Loading prepared raw datasets for {ticker_symbvol} on {exchange} ...")
    
    dfs:list[pd.DataFrame] = []

    for minute_multiplier in cnts.minute_multipliers.keys():
        dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
        df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
        if df is not None:
            dfs.append(df)

    return dfs