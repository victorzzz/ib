import pandas as pd
import numpy as np

total_seconds_from_930_to_1600 = 6 * 3600 + 30 * 60  # 6 hours and 30 minutes in seconds

# Create a DataFrame with a datetime column
df = pd.DataFrame({'timestamp': pd.date_range(start='2020-12-31 15:00', periods=1700000, freq='1S')})

# Original timestamps
print("Original timestamps:")
print(df)

# Normalize timestamps to midnight
df['timestamp_dt_toronto'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Toronto')

normalized_dt = df['timestamp_dt_toronto'].dt.normalize()
time_since_930 = (df['timestamp_dt_toronto'] - normalized_dt) - pd.Timedelta(hours=9, minutes=30)

# Normalized calculations
df['day_of_week'] = df['timestamp'].dt.dayofweek / 6  # Normalize: day 0 -> 0, day 6 -> 1
df['week'] = (df['timestamp'].dt.isocalendar().week - 1) / 52  # Normalize: week 1 -> 0, week 53 -> 1
df['day_of_year'] = (df['timestamp'].dt.dayofyear - 1) / 365  # Normalize: day 1 -> 0, day 366 -> 1

df['normalized_seconds_since_930'] = time_since_930.dt.total_seconds() / total_seconds_from_930_to_1600

print("\nTimestamps normalized to midnight:")
print(df)