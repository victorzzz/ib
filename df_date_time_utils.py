import logging
import numpy as np
import date_time_utils as dt_utils
import pandas as pd

total_seconds_from_930_to_1600 = 6 * 3600 + 30 * 60  # 6 hours and 30 minutes in seconds

"""
def add_normalized_time_columns(df:pd.DataFrame) -> pd.DataFrame:
    timestamp_column = df['timestamp']

    logging.info(f"START Adding normalized time columns ...")

    df['normalized_trading_time'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_float_with_round_6)
    df['normalized_week'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_week_with_round_3)
    df['normalized_day_of_week'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_day_of_week_with_round_2)
    df['normalized_day_of_year'] = timestamp_column.apply(dt_utils.nyse_timestamp_to_normalized_day_year_with_round_5)

    logging.info(f"END Adding normalized time columns ...")

    df = df.copy()

    logging.info(f"Adding normalized time columns ... DONE")

    return df
"""

def add_normalized_time_columns(df:pd.DataFrame) -> pd.DataFrame:
    # Normalize timestamps to midnight
    df['timestamp_dt_toronto'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Toronto')

    normalized_dt = df['timestamp_dt_toronto'].dt.normalize()
    time_since_930 = (df['timestamp_dt_toronto'] - normalized_dt) - pd.Timedelta(hours=9, minutes=30)

    # Normalized calculations
    df['normalized_day_of_week'] = normalized_dt.dt.dayofweek / 6  # Normalize: day 0 -> 0, day 6 -> 1
    df['normalized_week'] = (normalized_dt.dt.isocalendar().week - 1) / 52  # Normalize: week 1 -> 0, week 53 -> 1
    df['normalized_day_of_year'] = (normalized_dt.dt.dayofyear - 1) / 365  # Normalize: day 1 -> 0, day 366 -> 1

    df['normalized_trading_time'] = time_since_930.dt.total_seconds() / total_seconds_from_930_to_1600

    df = df.copy()

    return df
