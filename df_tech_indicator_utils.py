import pandas as pd
import numpy as np

"""
import ta
import ta.trend
import ta.momentum
import ta.volatility
import ta.volume
"""

import pandas_ta as ta

import logging

volume_profile_depths = (112, 224,)
depth_to_bins_koeff = 8

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

def add_volume_profile(df:pd.DataFrame) -> pd.DataFrame:

    print(df)
    print(df.index)

    vwap = df['TRADES_average'].to_numpy(copy=True)
    volume = df['TRADES_volume'].to_numpy(copy=True)

    total_records = df.shape[0]

    for depth in volume_profile_depths:

        logging.info(f"Processing volume profile for depth {depth} ...")

        num_bins = int(depth / depth_to_bins_koeff)
        df[f'vp_{depth}_width'] = 0

        for bin in range(num_bins):
            df[f'vp_{depth}_{bin}_price'] = 0.0
            df[f'vp_{depth}_{bin}_volume'] = 0.0

        df = df.copy()

        for index in range(depth, total_records):
          
          vwap_for_volume_profile = vwap[index - depth:index]
          volume_for_volume_profile = volume[index - depth:index]

          # if np.isnan(vwap_for_volume_profile).any() or np.isnan(volume_for_volume_profile).any():
            # logging.error(f"Nan values found for index {index}. Depth: {depth}. Total records: {total_records}. ")

          hist, bins = np.histogram(vwap_for_volume_profile, bins=num_bins, weights=volume_for_volume_profile)

          sum_hist = np.sum(hist)
          if (sum_hist != 0):
            hist = hist / sum_hist
          else:
              logging.warning(f"Sum of hist is 0 for index {index}. Depth: {depth}. Total records: {total_records}. ")

          sorted_indices = np.argsort(hist)[::-1]
          sorted_hist = hist[sorted_indices]
          sorted_bins_start = bins[:-1][sorted_indices]
          
          df.loc[index, f'vp_{depth}_width'] = bins[1] - bins[0]

          for histogram_index, item in enumerate(zip(sorted_bins_start, sorted_hist)):
              bin_start, histogram_volume = item

              df.loc[index, f'vp_{depth}_{histogram_index}_price'] = bin_start
              df.loc[index, f'vp_{depth}_{histogram_index}_volume'] = histogram_volume

    return df

