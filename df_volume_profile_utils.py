import pandas as pd
import numpy as np
import torch
import torchist as th

import logging

DEFAULT_VOLUME_PROFILE_DEPTHS = (128, 256, 512)
DEFAULT_VOLUME_TO_BIN_COEFF:float = 8.0
DEFAULT_TOP_BINS_COEFF:float = 4.0

def calculate_top_of_volume_profile(
    vwap:np.ndarray, 
    volume:np.ndarray, 
    depth:int, 
    num_bins:int, 
    top_bins:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    total_records = vwap.shape[0]
    hist_result = np.zeros((total_records, top_bins))
    bins_result = np.zeros((total_records, top_bins))
    widths = np.zeros(total_records)

    for index in range(depth, total_records):
        
        r_index = index - depth

        vwap_for_volume_profile = vwap[r_index:index]
        volume_for_volume_profile = volume[r_index:index]

        try:
            hist, bins = np.histogram(vwap_for_volume_profile, bins=num_bins, weights=volume_for_volume_profile, density=False)
            sum_hist = np.sum(hist)
            if sum_hist.item() != 0.0:
                hist = hist / sum_hist
            
            # Find the indices of the top 'top_bins' values in 'hist'
            top_indices = np.argpartition(hist, -top_bins)[-top_bins:]
            
            # Sort 'top_indices' by the 'hist' values in descending order
            top_indices_sorted = top_indices[np.argsort(hist[top_indices])[::-1]]
            
            for i, ti in enumerate(top_indices_sorted):
                hist_result[index, i] = hist[ti]
                bins_result[index, i] = bins[ti]
            
            widths[index] = bins[1] - bins[0]
        except Exception as e:
            logging.error(f"Error calculating volume profile for index {index}: {e}")
            raise e
        
    return (hist_result, bins_result, widths)

def add_top_of_volume_profile(
    df:pd.DataFrame, 
    price_column:str = 'TRADES_average', 
    volume_column:str = 'TRADES_volume', 
    volume_profile_depths:tuple[int, ...] = DEFAULT_VOLUME_PROFILE_DEPTHS, 
    depth_to_bins_coeff:float = DEFAULT_VOLUME_TO_BIN_COEFF,
    top_bins_coeff:float = DEFAULT_TOP_BINS_COEFF) -> pd.DataFrame:

    price = df[price_column].to_numpy(copy=True, dtype=np.float32)
    volume = df[volume_column].to_numpy(copy=True, dtype=np.float32)

    for depth in volume_profile_depths:

        logging.info(f"Processing volume profile for depth {depth} ...")

        num_bins = round(depth / depth_to_bins_coeff)
        top_bins = round(num_bins / top_bins_coeff)
        price_fileds = [f'vp_{depth}_{histogram_index}_price' for histogram_index in range(top_bins)]
        volume_fields = [f'vp_{depth}_{histogram_index}_volume' for histogram_index in range(top_bins)] 

        hist, bins, widths = calculate_top_of_volume_profile(price, volume, depth, num_bins, top_bins)

        df[f'vp_{depth}_width'] = widths
        df[price_fileds] = bins
        df[volume_fields] = hist

        df = df.copy()

    return df

#wvap = np.array([1, 20, 2, 30, 3, 40, 4, 50, 5, 60, 6, 70, 7, 80, 9, 3, 9, 4, 10, 5, 11, 9, 7, 6], dtype=np.float32)
#wvol = np.array([1, 2,   3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 11, 19, 20, 1, 2, 90023400000, 100345345], dtype=np.float32)
# wvap = np.array([10, 20, 30, 40, 200, 3, 4, 5, 6, 10, 10, 2, 3, 4, 10, 10, 1, 2, 3, 40])
# wvol = np.zeros(wvap.shape[0])

# print(calculate_volume_profile(wvap, wvol, 10, 3))

#dataframe:pd.DataFrame = pd.DataFrame({'TRADES_average': wvap, 'TRADES_volume': wvol})
# print(add_volume_profile(dataframe, volume_profile_depths=(12, 6), depth_to_bins_coeff=2))

#print(add_top_of_volume_profile(dataframe, volume_profile_depths=(12, 6), depth_to_bins_coeff=2, top_bins_coeff=2))

# hist, bins = np.histogram(wvap, bins=5, weights=wvol, density=False)
# hist1 = hist / np.sum(hist)

# print (hist1) 
# print (bins)
