import pandas as pd
import btalib as btalib

def add_tech_indicators(df:pd.DataFrame):
    # EMA 20, 50, 200 for Close and average trade price
    df['ema20'] = btalib.ema(df['Close'], period=20).df
