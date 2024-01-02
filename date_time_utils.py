from typing import Union
from datetime import time
from datetime import datetime
from datetime import date
import pytz


# Define NYSE timezone
nyse_timezone = pytz.timezone("America/New_York")
total_seconds = (16 - 9.5) * 60 * 60
start_time = time(hour=9, minute=30)

def nyse_timestamp_to_date_time(epoch_time:int) -> datetime:

    # Convert epoch time to a datetime object in the specified timezone
    dt = datetime.fromtimestamp(epoch_time, nyse_timezone)

    return dt

def nyse_timestamp_to_float_with_round_6(epoch_time:int) -> float:
    return round(nyse_timestamp_to_float(epoch_time), 6)

def nyse_timestamp_to_float(epoch_time:int) -> float:

    # Convert epoch time to a datetime object in the specified timezone
    dt = datetime.fromtimestamp(epoch_time, nyse_timezone)
    
    # Extract the date from the datetime object
    date = dt.date()
    
    # Combine the extracted date with the desired start and end times
    start_datetime = datetime.combine(date, start_time)
    
    # Calculate the elapsed seconds from the start time to the given epoch time
    elapsed_seconds = epoch_time - start_datetime.timestamp()
    
    # Calculate the float value based on the elapsed seconds and total seconds
    float_value = elapsed_seconds / total_seconds
    
    return float_value

def nyse_timestamp_to_normalized_week_with_round_3(epoch_time:int) -> float:
    
    week = nyse_timestamp_to_week(epoch_time) - 1

    normalized_week = week / 51.0

    return round(normalized_week, 3)


def nyse_timestamp_to_week(epoch_time:int) -> int:

    dt:datetime = nyse_timestamp_to_date_time(epoch_time)

    week_number = dt.isocalendar()[1]

    return week_number


def nyse_timestamp_to_normalized_day_of_week_with_round_2(epoch_time:int) -> float:

    day_of_week = nyse_timestamp_to_day_of_week(epoch_time) - 1
    
    normalized_day_of_week = day_of_week / 6.0

    return round(normalized_day_of_week, 2)


def nyse_timestamp_to_day_of_week(epoch_time:int) -> int:

    dt:datetime = nyse_timestamp_to_date_time(epoch_time)

    day_of_week = dt.isocalendar()[2]

    return day_of_week

def bar_date_to_epoch_time(bar_date:Union[date, datetime]) -> int:
    
    if isinstance(bar_date, datetime):
        # It's a datetime object
        return int(bar_date.timestamp())
    elif isinstance(bar_date, date):
        # It's a date object, convert to datetime by assuming midnight
        return int(datetime.combine(bar_date, time.min).timestamp())


