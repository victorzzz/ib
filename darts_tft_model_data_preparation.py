import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    #'1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
    '1m_MIDPOINT_close',
    #'1m_BID_open', 
    #'1m_BID_high', '1m_BID_low', 
    #'1m_BID_close',
    #'1m_ASK_open', 
    #'1m_ASK_high', '1m_ASK_low', 
    #'1m_ASK_close',
    '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
    '1m_TRADES_volume', '1m_TRADES_average',

    # '1m__t_MFI_TRADES_average_14',
    # '1m__t_RSI_TRADES_average_14',
    # '1m__t_CCI_TRADES_average_14',
    # '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3',
    
    # '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20',
    # '1m__t_BBP_TRADES_average_20',
    
    # '1m_vp_64_width',
    # '1m_vp_64_0_price', '1m_vp_64_1_price',
    # '1m_vp_64_0_volume', '1m_vp_64_1_volume',
    
    # '1m_vp_128_width',
    # '1m_vp_128_0_price', '1m_vp_128_1_price', '1m_vp_128_2_price', '1m_vp_128_3_price',
    # '1m_vp_128_0_volume', '1m_vp_128_1_volume', '1m_vp_128_2_volume', '1m_vp_128_3_volume',       
        
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

COVARIATE_PRICE_SCALING_COMPONENT_MASK:list[bool] = [x in PRICE_SCALING_COLUMNS for x in COVARIATE_COLUMNS]
COVARIATE_VOLUME_SCALING_COMPONENT_MASK:list[bool] = [x in VOLUME_SCALING_COLUMNS for x in COVARIATE_COLUMNS]

TARGET_PRICE_SCALING_COMPONENT_MASK:list[bool] = [x in PRICE_SCALING_COLUMNS for x in TARGET_COLUMNS]
TARGET_VOLUME_SCALING_COMPONENT_MASK:list[bool] = [x in VOLUME_SCALING_COLUMNS for x in TARGET_COLUMNS]

COVARIATE_PRICE_FITTING_COMPONENT_MASK:list[bool] = [x == PRICE_FITTING_COLUMN for x in COVARIATE_COLUMNS]
COVARIATE_VOLUME_FITTING_COMPONENT_MASK:list[bool] = [x == VOLUME_FITTING_COLUMN for x in COVARIATE_COLUMNS]

# returns target(train, val, test), covariate(train, val, test), future_covariate(train, val, test), price_scaler, volume_scaler
def prepare_train_val_test_datasets(symbol:str, exchange:str, train_part:float = 0.8, tail:float | None = None, generate_test:bool = False) -> \
        tuple[tuple[TimeSeries, TimeSeries, TimeSeries | None], tuple[TimeSeries, TimeSeries, TimeSeries | None], tuple[TimeSeries, TimeSeries, TimeSeries | None], Scaler, Scaler]:
    
    target_ts, covariate_ts, future_covariate_ts = prepare_timeseries_for_symbol(symbol, exchange, tail)

    price_scaler:Scaler = Scaler(name='price_scaler', scaler=StandardScaler())
    volume_scaler:Scaler = Scaler(name='volume_scaler', scaler=StandardScaler())
    
    # Split the time series
    train_target, val_test_target = target_ts.split_after(train_part)
    if generate_test:
        val_target, test_target = val_test_target.split_after(0.5)
    else:
        val_target = val_test_target
        test_target = None
    
    train_covariate, val_test_covariate = covariate_ts.split_after(train_part)
    if generate_test:
        val_covariate, test_covariate = val_test_covariate.split_after(0.5)
    else:
        val_covariate = val_test_covariate
        test_covariate = None
    
    train_future_covariate, val_test_future_covariate = future_covariate_ts.split_after(train_part)
    if generate_test:
        val_future_covariate, test_future_covariate = val_test_future_covariate.split_after(0.5)
    else:
        val_future_covariate = val_test_future_covariate
        test_future_covariate = None
    
    covariate_price_fitting_component_mask = np.array(COVARIATE_PRICE_FITTING_COMPONENT_MASK)
    covariate_volume_fitting_component_mask = np.array(COVARIATE_VOLUME_FITTING_COMPONENT_MASK)
    
    price_scaler.fit(train_covariate, component_mask=covariate_price_fitting_component_mask)
    volume_scaler.fit(train_covariate, component_mask=covariate_volume_fitting_component_mask)
    
    train_target = transform_ts_with_components_mask(train_target, price_scaler, TARGET_PRICE_SCALING_COMPONENT_MASK)
    train_target = transform_ts_with_components_mask(train_target, volume_scaler, TARGET_VOLUME_SCALING_COMPONENT_MASK)
    
    val_target = transform_ts_with_components_mask(val_target, price_scaler, TARGET_PRICE_SCALING_COMPONENT_MASK)
    val_target = transform_ts_with_components_mask(val_target, volume_scaler, TARGET_VOLUME_SCALING_COMPONENT_MASK)
    
    if test_target is not None:
        test_target = transform_ts_with_components_mask(test_target, price_scaler, TARGET_PRICE_SCALING_COMPONENT_MASK)
        test_target = transform_ts_with_components_mask(test_target, volume_scaler, TARGET_VOLUME_SCALING_COMPONENT_MASK)
    
    train_covariate = transform_ts_with_components_mask(train_covariate, price_scaler, COVARIATE_PRICE_SCALING_COMPONENT_MASK)
    train_covariate = transform_ts_with_components_mask(train_covariate, volume_scaler, COVARIATE_VOLUME_SCALING_COMPONENT_MASK)
    
    val_covariate = transform_ts_with_components_mask(val_covariate, price_scaler, COVARIATE_PRICE_SCALING_COMPONENT_MASK)
    val_covariate = transform_ts_with_components_mask(val_covariate, volume_scaler, COVARIATE_VOLUME_SCALING_COMPONENT_MASK)
    
    if test_covariate is not None:
        test_covariate = transform_ts_with_components_mask(test_covariate, price_scaler, COVARIATE_PRICE_SCALING_COMPONENT_MASK)
        test_covariate = transform_ts_with_components_mask(test_covariate, volume_scaler, COVARIATE_VOLUME_SCALING_COMPONENT_MASK)
    
    return (train_target, val_target, test_target), (train_covariate, val_covariate, test_covariate), (train_future_covariate, val_future_covariate, test_future_covariate), price_scaler, volume_scaler

def inverse_transform(ts:TimeSeries | None, priceScaler:Scaler, volumeScaler:Scaler) -> TimeSeries:
    
    if isinstance(ts, TimeSeries):    
        ts = inverse_transform_ts_with_components_mask(ts, priceScaler, TARGET_PRICE_SCALING_COMPONENT_MASK)
        ts = inverse_transform_ts_with_components_mask(ts, volumeScaler, TARGET_VOLUME_SCALING_COMPONENT_MASK)
    else:
        raise ValueError("The transformed time series is not a TimeSeries object.")
            
    return ts

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
    df_1m[COVARIATE_VOLUME_SCALING_COLUMNS] = np.log(df_1m[COVARIATE_VOLUME_SCALING_COLUMNS])
    df_1m.reset_index(drop=True, inplace=True)
    
    df_1m = df_1m.copy()
    
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

def transform_ts_with_components_mask(ts:TimeSeries, scaler:Scaler, component_mask_list:list[bool]) -> TimeSeries:
    masks:list[list[bool]] = transform_boolean_list(component_mask_list)
    
    for mask in masks:
        mask_np:np.ndarray = np.array(mask)
        transofrmed_ts:TimeSeries | list[TimeSeries] = scaler.transform(ts, component_mask=mask_np)
                
        if isinstance(transofrmed_ts, TimeSeries):
            ts = transofrmed_ts
        else:
            raise ValueError("The transformed time series is not a TimeSeries object.")
        
    return ts

def inverse_transform_ts_with_components_mask(ts:TimeSeries, scaler:Scaler, component_mask_list:list[bool]) -> TimeSeries:
    masks:list[list[bool]] = transform_boolean_list(component_mask_list)
    
    for mask in masks:
        mask_np:np.ndarray = np.array(mask)
        transofrmed_ts:TimeSeries | list[TimeSeries] | list[list[TimeSeries]] = scaler.inverse_transform(ts, component_mask=mask_np)
        
        if isinstance(transofrmed_ts, TimeSeries):
            ts = transofrmed_ts
        else:
            raise ValueError("The transformed time series is not a TimeSeries object.")
        
    return ts
    
def transform_boolean_list(original_list) -> list[list[bool]]:
    # Initialize an empty list to hold the transformed lists
    generated_list = []

    # Iterate over the original list to find the indices of True values
    for index, value in enumerate(original_list):
        if value:
            # Create a new list with the same length as original_list, filled with False
            new_list = [False] * len(original_list)
            # Set the current index to True
            new_list[index] = True
            # Add the new list to the generated_list
            generated_list.append(new_list)

    return generated_list

"""
data = pd.DataFrame({
    'time': pd.date_range(start='2022-01-01', periods=7, freq='D'),
    'Component1': [1, 2, 3, 4, 7, 8, 9],
    'Component2': [5, 6, 7, 8, 7, 8, 9],
    'Component3': [9, 10, 11, 12, 7, 8, 9],
    'Component4': [19, 110, 111, 112, 7, 8, 9],
    'Component5': [191, 1101, 1111, 1121, 7, 8, 9],
})

data.set_index('time', inplace=True)

scaler1 = Scaler(name='scaler', scaler=StandardScaler(), global_fit=True, verbose=True)
# scaler2 = Scaler()

ts:TimeSeries = TimeSeries.from_dataframe(data, value_cols=['Component1', 'Component2', 'Component3', 'Component4', 'Component5'])

scaler1.fit(ts, component_mask=np.array([True, False, False, False, False]))
# scaler.fit(ts)

ts_scaled = scaler1.transform(ts, component_mask=np.array([False, False, False, False, True]))
ts_scaled = scaler1.transform(ts_scaled, component_mask=np.array([True, False, False, False, False]))
ts_scaled = scaler1.transform(ts_scaled, component_mask=np.array([False, True, False, False, False]))

print(ts_scaled)
"""