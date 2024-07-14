import pandas as pd
import pandas_ta as ta

import logging

DEFAULT_PERIODS_1:tuple[int, ...] = (7, 14, 21)
DEFAULT_PERIODS_2:tuple[int, ...] = (20, 30)
DEFAULT_PERIODS_3:tuple[int, ...] = (13, 26)
DEFAULT_PERIODS_4:tuple[tuple[int, int, int], ...] = ((26, 12, 9), (39, 18, 13))
DEFAULT_PERIODS_5:tuple[tuple[int, int], ...] = ((14, 3), (21,4))

# DEFAULT_EMA_PERIODS:tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)

def add_ema(
    df:pd.DataFrame, 
    columns:list[str],
    periods:tuple[int, ...],
    add_ema_colums_to_df:bool,
    add_ema_dif_columns_to_df:bool,
    add_ema_retio_columns_to_df:bool) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if len(columns) == 0 or len(periods) == 0 or (not add_ema_colums_to_df and not add_ema_dif_columns_to_df and not add_ema_retio_columns_to_df):
        logging.error(f"Columns or periods are empty or no columns to add to dataframe")
        return (df, [], [], [])
    
    new_columns_ema = []
    new_columns_difs = []
    new_columns_ratios = []
    
    for period in periods:

        for column in columns:
            new_column_ema = f'_t_EMA_{column}_{period}'
            ema = ta.ema(df[column], length=period)
            
            if not isinstance(ema, pd.Series):
                logging.error(f"EMA for column {column} and period {period} is not a pandas series")
                continue
            
            if add_ema_colums_to_df:
                df[new_column_ema] = ema
                new_columns_ema.append(new_column_ema)
            
            if add_ema_dif_columns_to_df:
                new_column_dif = f'_t_EMA_dif_{column}_{period}'
                df[new_column_dif] = df[column] - ema
                new_columns_difs.append(new_column_dif)
                
            if add_ema_retio_columns_to_df:
                new_column_ratio = f'_t_EMA_ratio_{column}_{period}'
                df[new_column_ratio] = (df[column] / ema) - 1.0
                new_columns_ratios.append(new_column_ratio)
        
    return (df, new_columns_ema, new_columns_difs, new_columns_ratios)

# returns a tuple with:
#  - the dataframe
#  - the list colums need to be normalized with price normalizer
def add_technical_indicators(
    df:pd.DataFrame,
    time_frame_minute_multiplier:int,
    periods_1:tuple[int, ...] = DEFAULT_PERIODS_1,
    periods_2:tuple[int, ...] = DEFAULT_PERIODS_2,
    periods_3:tuple[int, ...] = DEFAULT_PERIODS_3,
    periods_4:tuple[tuple[int, int, int], ...] = DEFAULT_PERIODS_4,
    periods_5:tuple[tuple[int, int], ...] = DEFAULT_PERIODS_5,
    for_midpoint_price:bool = False,
    for_trades_average:bool = True
    ) -> tuple[pd.DataFrame, list[str], list[str]]:

    price_normalized_columns = []
    log_normalized_columns = []

    for period in periods_1:
        if for_midpoint_price:
            df[f'_t_MFI_MIDPOINT_{period}'] = ta.mfi(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
            df[f'_t_MFI_MIDPOINT_{period}'] = (df[f'_t_MFI_MIDPOINT_{period}'] - 50.0) / 50.0
        if for_trades_average:
            df[f'_t_MFI_TRADES_average_{period}'] = ta.mfi(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)
            df[f'_t_MFI_TRADES_average_{period}'] = (df[f'_t_MFI_TRADES_average_{period}'] - 50.0) / 50.0

    logging.info(f"Money flow index added to dataframe ...")    

    for period in periods_1:
        if for_midpoint_price:
            df[f'_t_CCI_MIDPOINT_{period}'] = ta.cci(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], length=period)
            df[f'_t_CCI_MIDPOINT_{period}'] = df[f'_t_CCI_MIDPOINT_{period}'] / 200.0
        if for_trades_average:
            df[f'_t_CCI_TRADES_average_{period}'] = ta.cci(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], length=period)
            df[f'_t_CCI_TRADES_average_{period}'] = df[f'_t_CCI_TRADES_average_{period}'] / 200.0
    
    logging.info(f"Commodity channel index added to dataframe ...")

    df = df.copy()
    logging.info(f"Dataframe copied 1 ...")

    """
    if for_midpoint_price:
        df['_t_ADI_MIDPOINT'] = ta.ad(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['_t_ADI_TRADES_average'] = ta.ad(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Accumulation distribution index added to dataframe ...")

    if for_midpoint_price:
        df['_t_OBV_MIDPOINT'] = ta.obv(df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['_t_OBV_TRADES_average'] = ta.obv(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"On balance volume added to dataframe ...")
    """

    if time_frame_minute_multiplier >= 390:
        for period in periods_2:
            if for_midpoint_price:
                df[f'_t_CMF_MIDPOINT_{period}'] = ta.cmf(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], df['MIDPOINT_open'], length=period)
            if for_trades_average:
                df[f'_t_CMF_TRADES_average_{period}'] = ta.cmf(df['TRADES_high'], df['TRADES_low'], df['TRADES_close'], df['TRADES_volume'], df['TRADES_open'], length=period)

        logging.info(f"Chaikin money flow added to dataframe ...")

        df = df.copy()
        logging.info(f"Dataframe copied 2 ...")
    

    # requires volume denominator
    for period in periods_3:
        if for_midpoint_price:
            df[f'_t_FI_MIDPOINT_{period}'] = ta.efi(df['MIDPOINT_close'], df['TRADES_volume'], length=period)
            log_normalized_columns.append(f'_t_FI_MIDPOINT_{period}')
        if for_trades_average:
            df[f'_t_FI_TRADES_average_{period}'] = ta.efi(df['TRADES_average'], df['TRADES_volume'], length=period)
            log_normalized_columns.append(f'_t_FI_TRADES_average_{period}')

    logging.info(f"Force index added to dataframe ...")

    # requires volume denominator
    if for_midpoint_price:
        df['_t_VPT_MIDPOINT'] = ta.pvt(df['MIDPOINT_close'], df['TRADES_volume'])
        log_normalized_columns.append('_t_VPT_MIDPOINT')
    if for_trades_average:
        df['_t_VPT_TRADES_average'] = ta.pvt(df['TRADES_average'], df['TRADES_volume'])
        log_normalized_columns.append('_t_VPT_TRADES_average')

    logging.info(f"Volume price trend added to dataframe ...")

    if for_midpoint_price:
        df['_t_NVI_MIDPOINT'] = ta.nvi(df['MIDPOINT_close'], df['TRADES_volume'])
    if for_trades_average:
        df['_t_NVI_TRADES_average'] = ta.nvi(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Negative volume index added to dataframe ...")
    
    df = df.copy()
    logging.info(f"Dataframe copied 3 ...")

    for period in periods_1:
        if for_midpoint_price:
            df[f'_t_RSI_MIDPOINT_{period}'] = ta.rsi(df['MIDPOINT_close'], length=period)
            df[f'_t_RSI_MIDPOINT_{period}'] = (df[f'_t_RSI_MIDPOINT_{period}'] - 50.0) / 50.0
        if for_trades_average:
            df[f'_t_RSI_TRADES_average_{period}'] = ta.rsi(df['TRADES_average'], length=period)
            df[f'_t_RSI_TRADES_average_{period}'] = (df[f'_t_RSI_TRADES_average_{period}'] - 50.0) / 50.0

    logging.info(f"Relative strength index added to dataframe ...")

    for period in periods_2:
        if for_midpoint_price:
            df[[f'_t_BBL_MIDPOINT_{period}', f'_t_BBM_MIDPOINT_{period}', f'_t_BBU_MIDPOINT_{period}', f'_t_BBB_MIDPOINT_{period}', f'_t_BBP_MIDPOINT_{period}']] = ta.bbands(df['MIDPOINT_close'], length=period)
            df[f'_t_BBB_MIDPOINT_{period}'] = df[f'_t_BBB_MIDPOINT_{period}'] / 100.0
            price_normalized_columns.append(
                [
                    f'_t_BBL_MIDPOINT_{period}', 
                    f'_t_BBM_MIDPOINT_{period}', 
                    f'_t_BBU_MIDPOINT_{period}', 
                ]
            )
        if for_trades_average:
            df[[f'_t_BBL_TRADES_average_{period}', f'_t_BBM_TRADES_average_{period}', f'_t_BBU_TRADES_average_{period}', f'_t_BBB_TRADES_average_{period}', f'_t_BBP_TRADES_average_{period}']] = ta.bbands(df['TRADES_average'], length=period)
            df[f'_t_BBB_TRADES_average_{period}'] = df[f'_t_BBB_TRADES_average_{period}'] / 100.0
            price_normalized_columns.append(
                [
                    f'_t_BBL_TRADES_average_{period}', 
                    f'_t_BBM_TRADES_average_{period}', 
                    f'_t_BBU_TRADES_average_{period}', 
                ]
            )

    logging.info(f"Bollinger bands added to dataframe ...")

    for period, s in periods_5:
        if for_midpoint_price:
            df[[f'_t_STOCH_k_MIDPOINT_{period}_{s}', f'_t_STOCH_d_MIDPOINT_{period}_{s}']] = ta.stoch(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], k=period, d=s, smooth_k=s)
            df.loc[df[f'_t_STOCH_k_MIDPOINT_{period}_{s}'] > 200.0, f'_t_STOCH_k_MIDPOINT_{period}_{s}'] = 200.0
            df.loc[df[f'_t_STOCH_k_MIDPOINT_{period}_{s}'] < -200.0, f'_t_STOCH_k_MIDPOINT_{period}_{s}'] = -200.0
            df.loc[df[f'_t_STOCH_d_MIDPOINT_{period}_{s}'] > 200.0, f'_t_STOCH_d_MIDPOINT_{period}_{s}'] = 200.0
            df.loc[df[f'_t_STOCH_d_MIDPOINT_{period}_{s}'] < -200.0, f'_t_STOCH_d_MIDPOINT_{period}_{s}'] = -200.0
            df[f'_t_STOCH_k_MIDPOINT_{period}_{s}'] = df[f'_t_STOCH_k_MIDPOINT_{period}_{s}'] / 200.0
            df[f'_t_STOCH_d_MIDPOINT_{period}_{s}'] = df[f'_t_STOCH_d_MIDPOINT_{period}_{s}'] / 200.0
        if for_trades_average:
            df[[f'_t_STOCH_k_TRADES_average_{period}_{s}', f'_t_STOCH_d_TRADES_average_{period}_{s}']] = ta.stoch(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], k=period, d=s, smooth_k=s)
            df.loc[df[f'_t_STOCH_k_TRADES_average_{period}_{s}'] > 200.0, f'_t_STOCH_k_TRADES_average_{period}_{s}'] = 200.0
            df.loc[df[f'_t_STOCH_k_TRADES_average_{period}_{s}'] < -200.0, f'_t_STOCH_k_TRADES_average_{period}_{s}'] = -200.0
            df.loc[df[f'_t_STOCH_d_TRADES_average_{period}_{s}'] > 200.0, f'_t_STOCH_d_TRADES_average_{period}_{s}'] = 200.0
            df.loc[df[f'_t_STOCH_d_TRADES_average_{period}_{s}'] < -200.0, f'_t_STOCH_d_TRADES_average_{period}_{s}'] = -200.0
            df[f'_t_STOCH_k_TRADES_average_{period}_{s}'] = df[f'_t_STOCH_k_TRADES_average_{period}_{s}'] / 200.0
            df[f'_t_STOCH_d_TRADES_average_{period}_{s}'] = df[f'_t_STOCH_d_TRADES_average_{period}_{s}'] / 200.0

    logging.info(f"Stochastic oscillator added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 4 ...")

    for slow, fast, signal in periods_4:
            if for_midpoint_price:
                df[[
                f'_t_MACD_MIDPOINT_{slow}_{fast}_{signal}',
                f'_t_MACD_signal_MIDPOINT_{slow}_{fast}_{signal}',
                f'_t_MACD_histogram_MIDPOINT_{slow}_{fast}_{signal}']] = ta.macd(df['MIDPOINT_close'], slow=slow, fast=fast, signal=signal)
            if for_trades_average:
                df[[
                f'_t_MACD_TRADES_{slow}_{fast}_{signal}',
                f'_t_MACD_signal_TRADES_{slow}_{fast}_{signal}',
                f'_t_MACD_histogram_TRADES_{slow}_{fast}_{signal}']] = ta.macd(df['TRADES_average'], slow=slow, fast=fast, signal=signal)

    logging.info(f"Moving average convergence divergence added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 5 ...")
    
    return (df, price_normalized_columns, log_normalized_columns)
