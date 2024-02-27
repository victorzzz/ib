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

volume_profile_depths = (56, 112, 224,)
depth_to_bins_koeff = 4

def add_technical_indicators(df:pd.DataFrame) -> pd.DataFrame:

    logging.info(f"START add_technical_indicators  ...")

    midpoint_high_np = df['MIDPOINT_high'].to_numpy(copy=True)
    midpoint_low_np = df['MIDPOINT_low'].to_numpy(copy=True)
    midpoint_close_np = df['MIDPOINT_close'].to_numpy(copy=True)
    trades_high_np = df['TRADES_high'].to_numpy(copy=True)
    trades_low_np = df['TRADES_low'].to_numpy(copy=True)
    trades_average_np = df['TRADES_average'].to_numpy(copy=True)
    trades_volume_np = df['TRADES_volume'].to_numpy(copy=True)
    # ask_high_np = df['ASK_high'].to_numpy(copy=True)
    # bid_low_np = df['BID_low'].to_numpy(copy=True)
    
    logging.info(f"np collected ...")

    # add columns for MFI

    """
    for period in (7, 14, 21, 28):
        df[f'MFI_MIDPOINT_{period}'] = 0.0
        df[f'MFI_TRADES_average'] = 0.0
        # df['MFI_BID_low_ASK_high_MIDPOINT'] = 0.0
    """
        
    # logging.info(f"Money flow index columns with 0.0 added to dataframe ...")

    # df = df.copy()

    # logging.info(f"Copyed dataframe 1.0 ...")

    for period in (7, 14, 21, 28):
        df[f'MFI_MIDPOINT_{period}'] = ta.mfi(midpoint_high_np, midpoint_low_np, midpoint_close_np, trades_volume_np, length=period)
        df[f'MFI_TRADES_average_{period}'] = ta.mfi(trades_high_np, trades_low_np, trades_average_np, trades_volume_np, length=period)
        # df['MFI_BID_low_ASK_high_MIDPOINT'] = ta.volume.money_flow_index(ask_high_np, bid_low_np, midpoint_close_np, trades_volume_np, window=period)

    logging.info(f"Money flow index added to dataframe ...")    
    df = df.copy()
    logging.info(f"COPYED after Money flow index added to dataframe ...")    

    for period in (7, 14, 21, 28):
        df[f'MFI_MIDPOINT_{period}_2'] = ta.mfi(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], length=period)
        df[f'MFI_TRADES_average_{period}_2'] = ta.mfi(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], length=period)
        # df['MFI_BID_low_ASK_high_MIDPOINT'] = ta.volume.money_flow_index(ask_high_np, bid_low_np, midpoint_close_np, trades_volume_np, window=period)

    logging.info(f"Money flow index 2 added to dataframe ...")    

    df = df.copy()

    """
    logging.info(f"Copyed dataframe 1.1 ...")

    for period in (10, 20, 40):
        for constant in (0.007, 0.015, 0.03):
            df[f'CCI_MIDPOINT_{period}_{int(constant*1000)}'] = ta.trend.cci(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], window=period, constant=constant)
            df[f'CCI_TRADES_average_{period}_{int(constant*1000)}'] = ta.trend.cci(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], window=period, constant=constant)
            # df[f'CCI_BID_low_ASK_high_MIDPOINT_{period}_{int(constant*1000)}'] = ta.trend.cci(df['ASK_high'], df['BID_low'], df['MIDPOINT_close'], window=period, constant=constant)
    
    logging.info(f"Commodity channel index added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 2 ...")

    df['ADI_MIDPOINT'] = ta.volume.acc_dist_index(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'])
    df['ADI_TRADES_average'] = ta.volume.acc_dist_index(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'])

    df['OBV_MIDPOINT'] = ta.volume.on_balance_volume(df['MIDPOINT_close'], df['TRADES_volume'])
    df['OBV_TRADES_average'] = ta.volume.on_balance_volume(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Accumulation distribution index and on balance volume added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 3 ...")

    for period in (10, 20, 30, 40):
        df[f'CMF_MIDPOINT_{period}'] = ta.volume.chaikin_money_flow(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], df['TRADES_volume'], window=period)
        df[f'CMF_TRADES_average_{period}'] = ta.volume.chaikin_money_flow(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], df['TRADES_volume'], window=period)

    logging.info(f"Chaikin money flow added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 4 ...")

    for period in (6, 13, 26):
        df[f'FI_MIDPOINT_{period}'] = ta.volume.force_index(df['MIDPOINT_close'], df['TRADES_volume'], window=period)
        df[f'FI_TRADES_average_{period}'] = ta.volume.force_index(df['TRADES_average'], df['TRADES_volume'], window=period)

    logging.info(f"Force index added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 5 ...")

    df['VPT_MIDPOINT'] = ta.volume.volume_price_trend(df['MIDPOINT_close'], df['TRADES_volume'])
    df['VPT_TRADES_average'] = ta.volume.volume_price_trend(df['TRADES_average'], df['TRADES_volume'])

    df['NVI_MIDPOINT'] = ta.volume.negative_volume_index(df['MIDPOINT_close'], df['TRADES_volume'])
    df['NVI_TRADES_average'] = ta.volume.negative_volume_index(df['TRADES_average'], df['TRADES_volume'])

    logging.info(f"Volume price trend and negative volume index added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 6 ...")

    for period in (7, 14, 21, 28):
        df[f'EOM_MIDPOINT_{period}'] = ta.volume.ease_of_movement(df['MIDPOINT_high'], df['MIDPOINT_low'], df['TRADES_volume'], window=period)
        df[f'EOM_TRADES_{period}'] = ta.volume.ease_of_movement(df['TRADES_high'], df['TRADES_low'], df['TRADES_volume'], window=period)

    logging.info(f"Ease of movement added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 7 ...")

    for period in (7, 14, 21, 28):
        df[f'ATR_MIDPOINT_{period}'] = ta.volatility.average_true_range(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], window=period)
        df[f'ATR_TRADES_average_{period}'] = ta.volatility.average_true_range(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], window=period)
        # df[f'ATR_BID_low_ASK_high_TRADES_average_{period}'] = ta.volatility.average_true_range(df['ASK_high'], df['BID_low'], df['TRADES_average'], window=period)

    logging.info(f"Average true range added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 8 ...")

    for period in (7, 14, 21, 28):
        df[f'RSI_MIDPOINT_{period}'] = ta.momentum.rsi(df['MIDPOINT_close'], window=period)
        df[f'RSI_TRADES_average_{period}'] = ta.momentum.rsi(df['TRADES_average'], window=period)

    logging.info(f"Relative strength index added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 9 ...")

    for period in (10, 20, 40):
        df[f'BB_MAVG_MIDPOINT_{period}'] = ta.volatility.bollinger_mavg(df['MIDPOINT_close'], window=period)
        df[f'BB_MAVG_TRADES_average_{period}'] = ta.volatility.bollinger_mavg(df['TRADES_average'], window=period)
        for k in (2, 3):
            df[f'BB_HBAND_MIDPOINT_{period}_{k}'] = ta.volatility.bollinger_hband(df['MIDPOINT_close'], window=period, window_dev=k)
            df[f'BB_HBAND_TRADES_average_{period}_{k}'] = ta.volatility.bollinger_hband(df['TRADES_average'], window=period, window_dev=k)
            # df[f'BB_HBAND_ASK_high_{period}_{k}'] = ta.volatility.bollinger_hband(df['ASK_high'], window=period, window_dev=k)

            df[f'BB_LBAND_MIDPOINT_{period}_{k}'] = ta.volatility.bollinger_lband(df['MIDPOINT_close'], window=period, window_dev=k)
            df[f'BB_LBAND_TRADES_average_{period}_{k}'] = ta.volatility.bollinger_lband(df['TRADES_average'], window=period, window_dev=k)
            # df[f'BB_LBAND_BID_low_{period}_{k}'] = ta.volatility.bollinger_lband(df['BID_low'], window=period, window_dev=k)

    logging.info(f"Bollinger bands added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 10 ...")

    for period, s in ((7,2), (10,2), (14,3), (21,4), (28,6)):
        df[f'STOCH_MIDPOINT_{period}_{s}'] = ta.momentum.stoch(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], window=period, smooth_window=s)
        df[f'STOCH_TRADES_average_{period}_{s}'] = ta.momentum.stoch(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], window=period, smooth_window=s)
        # df[f'STOCH_BID_low_ASK_high_TRADES_average_{period}_{s}'] = ta.momentum.stoch(df['ASK_high'], df['BID_low'], df['TRADES_average'], window=period, smooth_window=s)

    logging.info(f"Stochastic oscillator added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 11 ...")

    for step in (0.01, 0.02, 0.03):
        for max_step in (0.1, 0.2, 0.3):
            df[f'PSAR_UP_MIDPOINT_{step}_{max_step}'] = ta.trend.psar_up(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], step=step, max_step=max_step)
            df[f'PSAR_UP_TRADES_average_{step}_{max_step}'] = ta.trend.psar_up(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], step=step, max_step=max_step)
            # df[f'PSAR_UP_BID_low_ASK_high_TRADES_average_{step}_{max_step}'] = ta.trend.psar_up(df['ASK_high'], df['BID_low'], df['TRADES_average'], step=step, max_step=max_step)

            df[f'PSAR_DOWN_MIDPOINT_{step}_{max_step}'] = ta.trend.psar_down(df['MIDPOINT_high'], df['MIDPOINT_low'], df['MIDPOINT_close'], step=step, max_step=max_step)
            df[f'PSAR_DOWN_TRADES_average_{step}_{max_step}'] = ta.trend.psar_down(df['TRADES_high'], df['TRADES_low'], df['TRADES_average'], step=step, max_step=max_step)
            # df[f'PSAR_DOWN_BID_low_ASK_high_TRADES_average_{step}_{max_step}'] = ta.trend.psar_down(df['ASK_high'], df['BID_low'], df['TRADES_average'], step=step, max_step=max_step)

    logging.info(f"Parabolic SAR added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 12 ...")

    for slow, fast, signal in ((20, 9, 7), (26, 12, 9), (39, 18, 13),  (52, 24, 18)):
            df[f'MACD_MIDPOINT_{slow}_{fast}'] = ta.trend.macd(df['MIDPOINT_close'], window_slow=slow, window_fast=fast)
            df[f'MACD_TRADES_average_{slow}_{fast}'] = ta.trend.macd(df['TRADES_average'], window_slow=slow, window_fast=fast)

            df[f'MACD_DIFF_MIDPOINT_{slow}_{fast}_{signal}'] = ta.trend.macd_diff(df['MIDPOINT_close'], window_slow=slow, window_fast=fast, window_sign=signal)
            df[f'MACD_DIFF_TRADES_average_{slow}_{fast}_{signal}'] = ta.trend.macd_diff(df['TRADES_average'], window_slow=slow, window_fast=fast, window_sign=signal)

            df[f'MACD_SIGNAL_MIDPOINT_{slow}_{fast}'] = ta.trend.macd_signal(df['MIDPOINT_close'], window_slow=slow, window_fast=fast)
            df[f'MACD_SIGNAL_TRADES_average_{slow}_{fast}'] = ta.trend.macd_signal(df['TRADES_average'], window_slow=slow, window_fast=fast)

    logging.info(f"Moving average convergence divergence added to dataframe ...")

    df = df.copy()

    logging.info(f"Copyed dataframe 13 ...")
    """

    return df

def add_volume_profile(df:pd.DataFrame) -> pd.DataFrame:
    vwap = df['TRADES_average'].to_numpy(copy=True)
    volume = df['TRADES_volume'].to_numpy(copy=True)

    total_records = df.shape[0]

    for depth in volume_profile_depths:

        num_bins = int(depth / depth_to_bins_koeff)
        df[f'vp_{depth}_width'] = 0

        for bin in range(num_bins):
            df[f'vp_{depth}_{bin}_price'] = 0
            df[f'vp_{depth}_{bin}_volume'] = 0

        df = df.copy()

        for index in range(depth, total_records):
          
          vwap_for_volume_profile = vwap[index - depth:index]
          volume_for_volume_profile = volume[index - depth:index]

          hist, bins = np.histogram(vwap_for_volume_profile, bins=num_bins, weights=volume_for_volume_profile)

          sum_hist = np.sum(hist)
          if (sum_hist != 0):
            hist = hist / sum_hist
          else:
              logging.warning(f"Sum of hist is 0 for index {index}. Depth: {depth}. Total records: {total_records}. ")

          sorted_indices = np.argsort(hist)[::-1]
          sorted_hist = hist[sorted_indices]
          sorted_bins_start = bins[:-1][sorted_indices]

          row = df.iloc[index] 
          
          row[f'vp_{depth}_width']= f'{bins[1] - bins[0]:.6e}'

          for histogram_index, item in enumerate(zip(sorted_bins_start, sorted_hist)):
              bin_start, histogram_volume = item

              row[f'vp_{depth}_{histogram_index}_price'] = f'{bin_start:.6e}'
              row[f'vp_{depth}_{histogram_index}_volume'] = '0.0' if histogram_volume == 0.0 else f'{histogram_volume:.6e}'

    return df

