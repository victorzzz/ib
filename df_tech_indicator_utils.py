import pandas as pd
import numpy as np

import pandas_ta as ta

import logging

volume_profile_depths = (112, 224,)
depth_to_bins_koeff = 4

def add_technical_indicators(df:pd.DataFrame) -> pd.DataFrame:

    for period in (7, 14, 21):
        df[f'MFI_MIDPOINT_{period}'] = ta.mfi(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        df[f'MFI_TRADES_average_{period}'] = ta.mfi(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)
        df['MFI_BID_low_ASK_high_MIDPOINT'] = ta.mfi(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], df['TRADES_volume'], window=period)

    logging.info(f"Money flow index added to dataframe ...")    

    for period in (7, 14, 21):
            df[f'CCI_MIDPOINT_{period}'] = ta.cci(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], length=period)
            df[f'CCI_TRADES_average_{period}'] = ta.cci(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], length=period)
            df[f'CCI_BID_low_ASK_high_MIDPOINT_{period}'] = ta.cci(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], length=period)
    
    logging.info(f"Commodity channel index added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 1 ...")

    df['ADI_MIDPOINT'] = ta.ad(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'])
    df['ADI_TRADES_average'] = ta.ad(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Accumulation distribution index added to dataframe ...")

    df['OBV_MIDPOINT'] = ta.obv(df['MIDPOINT_close'], df['TRADES_volume'])
    df['OBV_TRADES_average'] = ta.obv(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"On balance volume added to dataframe ...")

    for period in (10, 20, 30):
        df[f'CMF_MIDPOINT_{period}'] = ta.cmf(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        df[f'CMF_TRADES_average_{period}'] = ta.cmf(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)

    logging.info(f"Chaikin money flow added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 2 ...")

    for period in (6, 13, 26):
        df[f'FI_MIDPOINT_{period}'] = ta.efi(df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        df[f'FI_TRADES_average_{period}'] = ta.efi(df['TRADES_average'], df['TRADES_volume'], length=period)

    logging.info(f"Force index added to dataframe ...")

    df['VPT_MIDPOINT'] = ta.pvt(df['MIDPOINT_close'], df['TRADES_volume'])
    df['VPT_TRADES_average'] = ta.pvt(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Volume price trend added to dataframe ...")

    df['NVI_MIDPOINT'] = ta.nvi(df['MIDPOINT_close'], df['TRADES_volume'])
    df['NVI_TRADES_average'] = ta.nvi(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Negative volume index added to dataframe ...")

    for period in (7, 14, 21):
        df[f'EOM_MIDPOINT_{period}'] = ta.eom(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        df[f'EOM_TRADES_{period}'] = ta.eom(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)

    logging.info(f"Ease of movement added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 3 ...")

    for period in (7, 14, 21):
        df[f'ATR_MIDPOINT_{period}'] = ta.atr(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], length=period)
        df[f'ATR_TRADES_average_{period}'] = ta.atr(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], length=period)
        df[f'ATR_BID_low_ASK_high_MIDPOINT_close_{period}'] = ta.atr(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], length=period)

    logging.info(f"Average true range added to dataframe ...")

    for period in (7, 14, 21):
        df[f'RSI_MIDPOINT_{period}'] = ta.rsi(df['MIDPOINT_close'], length=period)
        df[f'RSI_TRADES_average_{period}'] = ta.rsi(df['TRADES_average'], length=period)

    logging.info(f"Relative strength index added to dataframe ...")

    for period in (10, 20, 30):
        df[[f'BBL_MIDPOINT_{period}', f'BBM_MIDPOINT_{period}', f'BBU_MIDPOINT_{period}', f'BBB_MIDPOINT_{period}', f'BBP_MIDPOINT_{period}']] = ta.bbands(df['MIDPOINT_close'], length=period)
        df[[f'BBL_TRADES_average_{period}', f'BBM_TRADES_average_{period}', f'BBU_TRADES_average_{period}', f'BBB_TRADES_average_{period}', f'BBP_MIDPOINT_{period}']] = ta.bbands(df['TRADES_average'], length=period)

    logging.info(f"Bollinger bands added to dataframe ...")

    for period, s in ((10,2), (14,3), (21,4)):
        df[[f'STOCH_k_MIDPOINT_{period}_{s}', f'STOCH_d_MIDPOINT_{period}_{s}']] = ta.stoch(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], k=period, d=s, smooth_k=s)
        df[[f'STOCH_k_TRADES_average_{period}_{s}', f'STOCH_d_TRADES_average_{period}_{s}']] = ta.stoch(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], k=period, d=s, smooth_k=s)
        df[[f'STOCH_k_BID_low_ASK_high_TRADES_average_{period}_{s}', f'STOCH_d_BID_low_ASK_high_TRADES_average_{period}_{s}']] = ta.stoch(df['ASK_high'], df['BID_low'], df['TRADES_average'], k=period, d=s, smooth_k=s)

    logging.info(f"Stochastic oscillator added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 4 ...")

    """
    df[[
        f'PSARl_MIDPOINT', 
        f'PSARs_MIDPOINT',
        f'PSARaf_MIDPOINT',
        f'PSARr_MIDPOINT']] = ta.psar(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'])

    logging.info(f"Parabolic SAR added to dataframe ...")
    """

    for slow, fast, signal in ((20, 9, 7), (26, 12, 9), (39, 18, 13)):
            df[[
            f'MACD_MIDPOINT_{slow}_{fast}_{signal}',
            f'MACD_signal_MIDPOINT_{slow}_{fast}_{signal}',
            f'MACD_histogram_MIDPOINT_{slow}_{fast}_{signal}']] = ta.macd(df['MIDPOINT_close'], slow=slow, fast=fast, signal=signal)

            df[[
            f'MACD_MIDPOINT_{slow}_{fast}_{signal}',
            f'MACD_signal_MIDPOINT_{slow}_{fast}_{signal}',
            f'MACD_histogram_MIDPOINT_{slow}_{fast}_{signal}']] = ta.macd(df['TRADES_average'], slow=slow, fast=fast, signal=signal)

    logging.info(f"Moving average convergence divergence added to dataframe ...")

    df = df.copy()

    logging.info(f"Dataframe copied 5 ...")

    return df

def calculate_volume_profile(vwap:np.ndarray, volume:np.ndarray, depth:int, num_bins:int) -> np.ndarray:

    total_records = vwap.shape[0]
    result = np.zeros((total_records, num_bins + 2))

    for index in range(depth, total_records):
        
        r_index = index - depth

        vwap_for_volume_profile = vwap[r_index:index]
        volume_for_volume_profile = volume[r_index:index]

        hist, bins = np.histogram(vwap_for_volume_profile, bins=num_bins, weights=volume_for_volume_profile, density=True)
        
        result[index, 0] = bins[0]
        result[index, 1] = bins[1] - bins[0]
        result[index, 2:] = hist

    np.nan_to_num(result, copy=False)

    return result

def add_volume_profile(df:pd.DataFrame) -> pd.DataFrame:

    vwap = df['TRADES_average'].to_numpy(copy=True)
    volume = df['TRADES_volume'].to_numpy(copy=True)

    for depth in volume_profile_depths:

        logging.info(f"Processing volume profile for depth {depth} ...")

        num_bins = int(depth / depth_to_bins_koeff)
        volume_fields = [f'vp_{depth}_{histogram_index}_volume' for histogram_index in range(num_bins)] 

        """
        df[f'vp_{depth}_min_price'] = 0.0
        df[f'vp_{depth}_width'] = 0.0

        df[volume_fields] = 0.0
        """

        vp = calculate_volume_profile(vwap, volume, depth, num_bins)

        df[f'vp_{depth}_min_price'] = vp[:, 0]
        df[f'vp_{depth}_width'] = vp[:, 1]
        df[volume_fields] = vp[:, 2:]

        df = df.copy()

    return df

# wvap = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 1, 1, 1, 2, 3, 4])
# wvol = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# wvap = np.array([10, 20, 30, 40, 200, 3, 4, 5, 6, 10, 10, 2, 3, 4, 10, 10, 1, 2, 3, 40])
# wvol = np.zeros(wvap.shape[0])

# print(calculate_volume_profile(wvap, wvol, 10, 5))