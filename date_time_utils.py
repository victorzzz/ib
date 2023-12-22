import datetime
import pytz

# Define NYSE timezone
nyse_timezone = pytz.timezone("America/New_York")
total_seconds = (16 - 9.5) * 60 * 60
start_time = datetime.time(hour=9, minute=30)

def nyse_msec_timestamp_to_date_time(msec_timestamp:int) -> datetime:
    epoch_time = msec_timestamp / 1000

    # Convert epoch time to a datetime object in the specified timezone
    dt = datetime.datetime.fromtimestamp(epoch_time, nyse_timezone)

    return dt

def nyse_msec_timestamp_to_float_with_round_6(msec_timestamp:int) -> float:
    return round(nyse_msec_timestamp_to_float(msec_timestamp), 6)

def nyse_msec_timestamp_to_float(msec_timestamp:int) -> float:
    
    epoch_time = msec_timestamp / 1000

    # Convert epoch time to a datetime object in the specified timezone
    dt = datetime.datetime.fromtimestamp(epoch_time, nyse_timezone)
    
    # Extract the date from the datetime object
    date = dt.date()
    
    # Combine the extracted date with the desired start and end times
    start_datetime = datetime.datetime.combine(date, start_time)
    
    # Calculate the elapsed seconds from the start time to the given epoch time
    elapsed_seconds = epoch_time - start_datetime.timestamp()
    
    # Calculate the float value based on the elapsed seconds and total seconds
    float_value = elapsed_seconds / total_seconds
    
    return float_value

def nyse_msec_timestamp_to_normalized_week_with_round_3(msec_timestamp:int) -> float:
    
    week = nyse_msec_timestamp_to_week(msec_timestamp) - 1

    normalized_week = week / 51.0

    return round(normalized_week, 3)


def nyse_msec_timestamp_to_week(msec_timestamp:int) -> int:

    dt = nyse_msec_timestamp_to_date_time(msec_timestamp)

    week_number = dt.isocalendar()[1]

    return week_number


def nyse_msec_timestamp_to_normalized_day_of_week_with_round_2(msec_timestamp:int) -> float:

    day_of_week = nyse_msec_timestamp_to_day_of_week(msec_timestamp) - 1
    
    normalized_day_of_week = day_of_week / 6.0

    return round(normalized_day_of_week, 2)


def nyse_msec_timestamp_to_day_of_week(msec_timestamp:int) -> int:

    dt = nyse_msec_timestamp_to_date_time(msec_timestamp)

    day_of_week = dt.isocalendar()[2]

    return day_of_week

