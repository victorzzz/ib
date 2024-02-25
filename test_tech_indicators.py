import pandas as pd
import ta
import ta.trend
import ta.momentum
import ta.volatility
import ta.volume

ind = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

df = pd.DataFrame({
    'indexf': ind,
    'close':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9]
    })
df.set_index('indexf', inplace=True)
print(df)

# Calculate the 14-period EMA of the closing prices
# df['EMA_14'] = ta.trend.ema_indicator(df['close'], window=5)

# Calculate the 50-period EMA of the closing prices
# df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=15)


print(df)

df2 = df.copy()

# Step 1: Reverse your DataFrame
df_reversed = df2.iloc[::-1].reset_index(drop=True)

# Step 2: Calculate the EMA on the reversed DataFrame
# For example, calculating a 14-period EMA
df_reversed['EMA_14'] = ta.trend.ema_indicator(df_reversed['close'], window=14,)

# Optional Step 3: Reverse the EMA column to match the original data order
df_reversed_back = df_reversed.iloc[::-1].reset_index(drop=True)

print(df_reversed_back)
