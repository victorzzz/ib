import date_time_utils as dt_utils
import pandas as pd

def add_normalized_time_columns(df:pd.DataFrame) -> pd.DataFrame:
    timestamp_column = df['timestamp']

    df['normalized_trading_time'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_float_with_round_6)
    df['normalized_week'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_week_with_round_3)
    df['normalized_day_of_week'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_day_of_week_with_round_2)
    df['normalized_day_of_year'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_day_year_with_round_5)

    df = df.copy()

    return df

