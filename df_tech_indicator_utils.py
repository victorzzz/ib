import pandas as pd
import numpy as np
import pandas_ta as ta

import logging

DEFAULT_PERIODS_1:tuple[int, int] = (14, 21)
DEFAULT_PERIODS_2:tuple[int, int] = (20, 30)
DEFAULT_PERIODS_3:tuple[int, int] = (13, 26)
DEFAULT_PERIODS_4:tuple[tuple[int, int, int], tuple[int, int, int]] = ((26, 12, 9), (39, 18, 13))
DEFAULT_PERIODS_5:tuple[tuple[int, int], tuple[int, int]] = ((14,3), (21,4))

def add_technical_indicators(
    df:pd.DataFrame,
    periods_1:tuple[int, int] = DEFAULT_PERIODS_1,
    periods_2:tuple[int, int] = DEFAULT_PERIODS_2,
    periods_3:tuple[int, int] = DEFAULT_PERIODS_3,
    periods_4:tuple[tuple[int, int, int], tuple[int, int, int]] = DEFAULT_PERIODS_4,
    periods_5:tuple[tuple[int, int], tuple[int, int]] = DEFAULT_PERIODS_5,
    for_midpoint_price:bool = False,
    for_trades_average:bool = True,
    for_bid_low_ask_high:bool = False
    ) -> pd.DataFrame:

    for period in periods_1:
        if for_midpoint_price:
            df[f'MFI_MIDPOINT_{period}'] = ta.mfi(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        if for_trades_average:
            df[f'MFI_TRADES_average_{period}'] = ta.mfi(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)   
        if for_bid_low_ask_high:
            df[f'MFI_BID_low_ASK_high_MIDPOINT_{period}'] = ta.mfi(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)    

    logging.info(f"Money flow index added to dataframe ...")    

    for period in periods_1:
        if for_midpoint_price:
            df[f'CCI_MIDPOINT_{period}'] = ta.cci(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], length=period)
        if for_trades_average:
            df[f'CCI_TRADES_average_{period}'] = ta.cci(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], length=period)
        if for_bid_low_ask_high:
            df[f'CCI_BID_low_ASK_high_MIDPOINT_{period}'] = ta.cci(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], length=period)
    
    logging.info(f"Commodity channel index added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 1 ...")

    if for_midpoint_price:
        df['ADI_MIDPOINT'] = ta.ad(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['ADI_TRADES_average'] = ta.ad(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Accumulation distribution index added to dataframe ...")

    if for_midpoint_price:
        df['OBV_MIDPOINT'] = ta.obv(df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['OBV_TRADES_average'] = ta.obv(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"On balance volume added to dataframe ...")

    for period in periods_2:
        if for_midpoint_price:
            df[f'CMF_MIDPOINT_{period}'] = ta.cmf(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        if for_trades_average:
            df[f'CMF_TRADES_average_{period}'] = ta.cmf(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)
        if for_bid_low_ask_high:
            df[f'CMF_BID_low_ASK_high_MIDPOINT_{period}'] = ta.cmf(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)

    logging.info(f"Chaikin money flow added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 2 ...")

    for period in periods_3:
        if for_midpoint_price:
            df[f'FI_MIDPOINT_{period}'] = ta.efi(df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        if for_trades_average:
            df[f'FI_TRADES_average_{period}'] = ta.efi(df['TRADES_average'], df['TRADES_volume'], length=period)

    logging.info(f"Force index added to dataframe ...")

    if for_midpoint_price:
        df['VPT_MIDPOINT'] = ta.pvt(df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['VPT_TRADES_average'] = ta.pvt(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Volume price trend added to dataframe ...")

    if for_midpoint_price:
        df['NVI_MIDPOINT'] = ta.nvi(df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['NVI_TRADES_average'] = ta.nvi(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Negative volume index added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 3 ...")

    for period in periods_1:
        if for_midpoint_price:
            df[f'ATR_MIDPOINT_{period}'] = ta.atr(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], length=period)
        if for_trades_average:
            df[f'ATR_TRADES_average_{period}'] = ta.atr(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], length=period)
        if for_bid_low_ask_high:
            df[f'ATR_BID_low_ASK_high_MIDPOINT_close_{period}'] = ta.atr(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], length=period)

    logging.info(f"Average true range added to dataframe ...")

    for period in periods_1:
        if for_midpoint_price:
            df[f'RSI_MIDPOINT_{period}'] = ta.rsi(df['MIDPOINT_close'], length=period)
        if for_trades_average:
            df[f'RSI_TRADES_average_{period}'] = ta.rsi(df['TRADES_average'], length=period)

    logging.info(f"Relative strength index added to dataframe ...")

    for period in periods_2:
        if for_midpoint_price:
            df[[f'BBL_MIDPOINT_{period}', f'BBM_MIDPOINT_{period}', f'BBU_MIDPOINT_{period}', f'BBB_MIDPOINT_{period}', f'BBP_MIDPOINT_{period}']] = ta.bbands(df['MIDPOINT_close'], length=period)
        if for_trades_average:
            df[[f'BBL_TRADES_average_{period}', f'BBM_TRADES_average_{period}', f'BBU_TRADES_average_{period}', f'BBB_TRADES_average_{period}', f'BBP_MIDPOINT_{period}']] = ta.bbands(df['TRADES_average'], length=period)

    logging.info(f"Bollinger bands added to dataframe ...")

    for period, s in periods_5:
        if for_midpoint_price:
            df[[f'STOCH_k_MIDPOINT_{period}_{s}', f'STOCH_d_MIDPOINT_{period}_{s}']] = ta.stoch(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], k=period, d=s, smooth_k=s)
        if for_trades_average:
            df[[f'STOCH_k_TRADES_average_{period}_{s}', f'STOCH_d_TRADES_average_{period}_{s}']] = ta.stoch(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], k=period, d=s, smooth_k=s)
        if for_bid_low_ask_high:
            df[[f'STOCH_k_BID_low_ASK_high_TRADES_average_{period}_{s}', f'STOCH_d_BID_low_ASK_high_TRADES_average_{period}_{s}']] = ta.stoch(df['ASK_high'], df['BID_low'], df['TRADES_average'], k=period, d=s, smooth_k=s)

    logging.info(f"Stochastic oscillator added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 4 ...")

    for slow, fast, signal in periods_4:
            if for_midpoint_price:
                df[[
                f'MACD_MIDPOINT_{slow}_{fast}_{signal}',
                f'MACD_signal_MIDPOINT_{slow}_{fast}_{signal}',
                f'MACD_histogram_MIDPOINT_{slow}_{fast}_{signal}']] = ta.macd(df['MIDPOINT_close'], slow=slow, fast=fast, signal=signal)
            if for_trades_average:
                df[[
                f'MACD_TRADES_{slow}_{fast}_{signal}',
                f'MACD_signal_TRADES_{slow}_{fast}_{signal}',
                f'MACD_histogram_TRADES_{slow}_{fast}_{signal}']] = ta.macd(df['TRADES_average'], slow=slow, fast=fast, signal=signal)

    logging.info(f"Moving average convergence divergence added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 5 ...")

    return df
