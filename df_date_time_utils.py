import constants as cnts
import pandas as pd
import numpy as np

def add_normalized_time_columns(df:pd.DataFrame):
    df['normalized_trading_time'] = df['timestamp'].apply(cnts.nyse_timestamp_to_float_with_round_6)

    # df = df[(df['normalized_trading_time'] >= 0.0) & (df['normalized_trading_time'] <= 1.0)]
    # df['vwap'].fillna( np.round((df['open'] + df['close']) / 2.0, 6), inplace=True)

    df['normalized_week'] = df['timestamp'].apply(cnts.nyse_timestamp_to_normalized_week_with_round_3)
    df['normalized_day_of_week'] = df['timestamp'].apply(cnts.nyse_timestamp_to_normalized_day_of_week_with_round_2)
    df['day_of_week'] = df['timestamp'].apply(cnts.nyse_timestamp_to_day_of_week)

