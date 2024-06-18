import logging

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import constants as cnts
import df_loader_saver as df_ls

TARGET_COLUMNS:list[str] = [
    '1m_BID_close', '1m_ASK_close',
    '1m_BID_high', '1m_BID_low',
    '1m_ASK_high', '1m_ASK_low',
    '1m_BID_open', '1m_ASK_open',
    ]

COVARIATE_COLUMNS:list[str] = [
    #'1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', '1m_MIDPOINT_close'
    #'1m_BID_open', 
    #'1m_BID_high', '1m_BID_low', 
    #'1m_BID_close',
    #'1m_ASK_open', 
    #'1m_ASK_high', '1m_ASK_low', 
    #'1m_ASK_close',
    '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
    '1m_TRADES_volume', '1m_TRADES_average',

    '1m__t_MFI_TRADES_average_14',
    '1m__t_RSI_TRADES_average_14',
    '1m__t_CCI_TRADES_average_14',
    '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3',
    
    '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20',
    '1m__t_BBP_TRADES_average_20',
    
    '1m_vp_64_width',
    '1m_vp_64_0_price', '1m_vp_64_1_price',
    '1m_vp_64_0_volume', '1m_vp_64_1_volume',
    
    '1m_vp_128_width',
    '1m_vp_128_0_price', '1m_vp_128_1_price', '1m_vp_128_2_price', '1m_vp_128_3_price',
    '1m_vp_128_0_volume', '1m_vp_128_1_volume', '1m_vp_128_2_volume', '1m_vp_128_3_volume',       
        
    # '1m__t_MFI_TRADES_average_7', '1m__t_MFI_TRADES_average_14', '1m__t_MFI_TRADES_average_21',
    # '1m__t_RSI_TRADES_average_7', '1m__t_RSI_TRADES_average_14', '1m__t_RSI_TRADES_average_21',
    # '1m__t_CCI_TRADES_average_7', '1m__t_CCI_TRADES_average_14', '1m__t_CCI_TRADES_average_21',
    # '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3', 
    # '1m__t_STOCH_k_TRADES_average_21_4', '1m__t_STOCH_d_TRADES_average_21_4',
    ]

FUTURE_COVARIATE_COLUMNS:list[str] = [
    '1m_normalized_day_of_week', '1m_normalized_week', '1m_normalized_trading_time'
    ]

PRICE_SCALING_COLUMNS:list[str] = [
    '1m_MIDPOINT_close', '1m_BID_close', '1m_ASK_close',
    '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low',
    '1m_BID_open', '1m_BID_high', '1m_BID_low',
    '1m_ASK_open', '1m_ASK_high', '1m_ASK_low',
    '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
    '1m_TRADES_average',
    '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20',
    '1m_vp_64_0_price', '1m_vp_64_1_price',
    '1m_vp_128_0_price', '1m_vp_128_1_price', '1m_vp_128_2_price', '1m_vp_128_3_price'
    ]

COVARIATE_PRICE_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x in PRICE_SCALING_COLUMNS]

PRICE_FITTING_COLUMN:str = '1m_MIDPOINT_close'

VOLUME_SCALING_COLUMNS:list[str] = ['1m_TRADES_volume']
VOLUME_FITTING_COLUMN:str = '1m_TRADES_volume'

COVARIATE_VOLUME_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x in VOLUME_SCALING_COLUMNS]

COVARIATE_NOT_SCALING_COLUMNS:list[str] = [x for x in COVARIATE_COLUMNS if x not in PRICE_SCALING_COLUMNS and x not in VOLUME_SCALING_COLUMNS]

COVARIATE_SCALING_COMPONENT_MASK:list[bool] = [x in VOLUME_SCALING_COLUMNS or x in PRICE_SCALING_COLUMNS for x in COVARIATE_COLUMNS]

# returns target(trane, val, test), covariate(train, val, test), future_covariate(train, val, test), price_scaler, volume_scaler
def prepare_traine_val_test_datasets(symbol:str, exchange:str, tail:float | None = None) -> \
        tuple[tuple[TimeSeries, TimeSeries, TimeSeries], tuple[TimeSeries, TimeSeries, TimeSeries], tuple[TimeSeries, TimeSeries, TimeSeries], Scaler, Scaler]:
    
    target_ts, covariate_ts, future_covariate_ts = prepare_timeseries_for_symbol(symbol, exchange, tail)

    target_scaler = Scaler(name='target_scaler')
    covariate_scaler = Scaler(name='covariate_scaler')
    
    # Split the time series
    train_target, val_test_target = target_ts.split_after(0.8)
    val_target, test_target = val_test_target.split_after(0.5)
    
    train_covariate, val_test_covariate = covariate_ts.split_after(0.8)
    val_covariate, test_covariate = val_test_covariate.split_after(0.5)
    
    train_future_covariate, val_test_future_covariate = future_covariate_ts.split_after(0.8)
    val_future_covariate, test_future_covariate = val_test_future_covariate.split_after(0.5)
    
    covariate_scaled_component_mask:np.ndarray = np.array(COVARIATE_SCALING_COMPONENT_MASK)
    
    target_scaler.fit(train_target)
    covariate_scaler.fit(train_covariate, component_mask=covariate_scaled_component_mask)
    
    train_target = target_scaler.transform(train_target)
    val_target = target_scaler.transform(val_target)
    test_target = target_scaler.transform(test_target)
    
    train_covariate = covariate_scaler.transform(train_covariate, component_mask=covariate_scaled_component_mask)
    val_covariate = covariate_scaler.transform(val_covariate, component_mask=covariate_scaled_component_mask)
    test_covariate = covariate_scaler.transform(test_covariate, component_mask=covariate_scaled_component_mask)
    
    if isinstance(train_target, TimeSeries) and isinstance(val_target, TimeSeries) and isinstance(test_target, TimeSeries) \
        and isinstance(train_covariate, TimeSeries) and isinstance(val_covariate, TimeSeries) and isinstance(test_covariate, TimeSeries) :
    
        return (train_target, val_target, test_target), (train_covariate, val_covariate, test_covariate), (train_future_covariate, val_future_covariate, test_future_covariate), target_scaler, covariate_scaler
    else:
        raise ValueError("One of the time series is not a TimeSeries object.")


# returns (target_timeseries, covariate_timeseries)
def prepare_timeseries_from_dataframe(df:pd.DataFrame) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    
    target_ts = TimeSeries.from_dataframe(df, value_cols = TARGET_COLUMNS)
    
    covariate_ts = TimeSeries.from_dataframe(df, value_cols = COVARIATE_COLUMNS)

    future_ts = TimeSeries.from_dataframe(df, value_cols = FUTURE_COVARIATE_COLUMNS)
    
    return target_ts, covariate_ts, future_ts

def prepare_timeseries_for_symbol(symbol:str, exchange:str, tail:float | None = None) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    df_1m:pd.DataFrame = load_prepared_raw_datasets(symbol, exchange)[0]
    
    if tail is not None:
        df_1m = df_1m.tail(int(len(df_1m) * tail))
    
    df_1m['1m_timestamp'] = df_1m['1m_timestamp'].astype('int64')
    df_1m.reset_index(drop=True, inplace=True)
    
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