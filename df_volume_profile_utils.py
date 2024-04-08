import pandas as pd
import numpy as np
import torch
import torchist as th

import logging

DEFAULT_VOLUME_PROFILE_DEPTHS = (112, 224,)
DEFAULT_VOLUME_TO_BIN_COEFF:float = 8.0
DEFAULT_TOP_BINS_COEFF:float = 4.0

def calculate_volume_profile(
    vwap_np_array:np.ndarray, 
    volume_np_array:np.ndarray, 
    depth:int, 
    num_bins:int) -> tuple[np.ndarray, np.ndarray]:
    
    # Convert your numpy arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vwap_tensor:torch.Tensor = torch.tensor(vwap_np_array, dtype=torch.float32, device=device)
    volume_tensor:torch.Tensor = torch.tensor(volume_np_array, dtype=torch.float32, device=device)

    hist, bins = calculate_volume_profile_torch(vwap_tensor, volume_tensor, depth, num_bins)
    
    # Remember to move the result back to CPU and convert to numpy if needed for further non-GPU computations
    hist_np:np.ndarray = hist.cpu().numpy()
    bins_np:np.ndarray = bins.cpu().numpy()
    
    return (hist_np, bins_np)

def calculate_volume_profile_torch(
    vwap: torch.Tensor, 
    volume: torch.Tensor, 
    depth: int, 
    num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    
    total_records = vwap.shape[0]
    
    # Initialize output tensors
    # Assuming `torch.histogram` returns `num_bins` values, and `num_bins + 1` bin edges
    
    histograms = torch.zeros((total_records, num_bins), device=vwap.device)
    bin_edges = torch.zeros((total_records, num_bins + 1), device=vwap.device)
    
    for index in range(depth, total_records):
        
        r_index:int = index - depth

        vwap_for_volume_profile = vwap[r_index:index]
        volume_for_volume_profile = volume[r_index:index]
    
        bins:torch.Tensor = th.histogram_edges(vwap_for_volume_profile, num_bins).to(vwap.device)
        hist:torch.Tensor = th.histogram(vwap_for_volume_profile, edges=bins, weights=volume_for_volume_profile)
    
        sum_hist:torch.Tensor = torch.sum(hist).float()  # ensure floating point division
        if sum_hist.item() != 0.0:
            hist = hist / sum_hist

        histograms[r_index] = hist
        bin_edges[r_index] = bins
    
    # Replace NaN values with 0. This operation is in-place.
    histograms = torch.nan_to_num(histograms)

    return (histograms, bin_edges)

def add_volume_profile(
    df:pd.DataFrame, 
    price_column:str = 'TRADES_average', 
    volume_column:str = 'TRADES_volume', 
    volume_profile_depths:tuple[int, ...] = DEFAULT_VOLUME_PROFILE_DEPTHS, 
    depth_to_bins_coeff:float = DEFAULT_VOLUME_TO_BIN_COEFF) -> pd.DataFrame:

    vwap = df[price_column].to_numpy(copy=True, dtype=np.float32)
    volume = df[volume_column].to_numpy(copy=True, dtype=np.float32)

    for depth in volume_profile_depths:

        logging.info(f"Processing volume profile for depth {depth} ...")

        num_bins = round(depth / depth_to_bins_coeff)
        volume_fields = [f'vp_{depth}_{histogram_index}_volume' for histogram_index in range(num_bins)] 

        hist, bins = calculate_volume_profile(vwap, volume, depth, num_bins)

        df[f'vp_{depth}_min_price'] = bins[:, 0]
        df[f'vp_{depth}_width'] = bins[:, 1] -  bins[:, 0]
        df[volume_fields] = hist

        df = df.copy()

    return df


# top of valume profile

def calculate_top_of_volume_profile(
    vwap_np_array:np.ndarray, 
    volume_np_array:np.ndarray, 
    depth:int, 
    num_bins:int,
    top_bins:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert your numpy arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vwap_tensor:torch.Tensor = torch.tensor(vwap_np_array, dtype=torch.float32, device=device)
    volume_tensor:torch.Tensor = torch.tensor(volume_np_array, dtype=torch.float32, device=device)

    hist, bins, bin_widths = calculate_top_of_volume_profile_torch(vwap_tensor, volume_tensor, depth, num_bins, top_bins)
    
    # Remember to move the result back to CPU and convert to numpy if needed for further non-GPU computations
    hist_np:np.ndarray = hist.cpu().numpy()
    bins_np:np.ndarray = bins.cpu().numpy()
    bin_widths_np:np.ndarray = bin_widths.cpu().numpy()
    
    return (hist_np, bins_np, bin_widths_np)

def calculate_top_of_volume_profile_torch(
    vwap: torch.Tensor, 
    volume: torch.Tensor, 
    depth: int, 
    num_bins: int,
    top_bins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    total_records = vwap.shape[0]
    
    # Initialize output tensors
    
    histograms = torch.zeros((total_records, top_bins), device=vwap.device)
    bin_edges = torch.zeros((total_records, top_bins), device=vwap.device)
    bin_widths = torch.zeros(total_records, device=vwap.device)
    
    for index in range(depth, total_records):
        
        r_index:int = index - depth

        vwap_for_volume_profile = vwap[r_index:index]
        volume_for_volume_profile = volume[r_index:index]
    
        bins:torch.Tensor = th.histogram_edges(vwap_for_volume_profile, num_bins).to(vwap.device)
        hist:torch.Tensor = th.histogram(vwap_for_volume_profile, edges=bins, weights=volume_for_volume_profile)

        bin_width = bins[1] - bins[0]
        
        bin_widths[r_index] = bin_width.item()
    
        sum_hist:torch.Tensor = torch.sum(hist).float()  # ensure floating point division
        if sum_hist.item() != 0.0:
            hist = hist / sum_hist

        # Sort `hist` in descending order and adjust `bins` accordingly
        sorted_hist, sorted_indices = torch.sort(hist, descending=True)
        
        # Only considering the first `top_bins`, adjust the bins accordingly
        # Since bins are edges, we pick the edges that correspond to the sorted top bins
        # Here, we pick the lower edge of each of the top bins for simplicity
        sorted_indices_for_edges = sorted_indices[:top_bins]
        
        # Populate the top histograms and their corresponding bin edges
        histograms[r_index] = sorted_hist[:top_bins]
        
        # Adjust for the bin edges; each bin's left edge is what we consider here
        bin_edges[r_index] = bins[sorted_indices_for_edges]
    
    # Replace NaN values with 0. This operation is in-place.
    histograms = torch.nan_to_num(histograms)

    return (histograms, bin_edges, bin_widths)

def add_top_of_volume_profile(
    df:pd.DataFrame, 
    price_column:str = 'TRADES_average', 
    volume_column:str = 'TRADES_volume', 
    volume_profile_depths:tuple[int, ...] = DEFAULT_VOLUME_PROFILE_DEPTHS, 
    depth_to_bins_coeff:float = DEFAULT_VOLUME_TO_BIN_COEFF,
    top_bins_coeff:float = DEFAULT_TOP_BINS_COEFF) -> pd.DataFrame:

    vwap = df[price_column].to_numpy(copy=True, dtype=np.float32)
    volume = df[volume_column].to_numpy(copy=True, dtype=np.float32)

    for depth in volume_profile_depths:

        logging.info(f"Processing volume profile for depth {depth} ...")

        num_bins = round(depth / depth_to_bins_coeff)
        top_bins = round(num_bins / top_bins_coeff)
        price_fileds = [f'vp_{depth}_{histogram_index}_price' for histogram_index in range(top_bins)]
        volume_fields = [f'vp_{depth}_{histogram_index}_volume' for histogram_index in range(top_bins)] 

        hist, bins, bin_widths = calculate_top_of_volume_profile(vwap, volume, depth, num_bins, top_bins)

        df[f'vp_{depth}_width'] = bin_widths
        df[price_fileds] = bins
        df[volume_fields] = hist

        df = df.copy()

    return df

wvap = np.array([1, 200, 1, 200, 1, 300, 1, 400, 1, 500, 1, 600, 1, 700, 1, 3, 1, 4, 1, 5, 1, 9, 7, 6], dtype=np.float32)
wvol = np.array([1, 2,   3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 11, 19, 20, 1, 2, 90023400000, 100345345], dtype=np.float32)
# wvap = np.array([10, 20, 30, 40, 200, 3, 4, 5, 6, 10, 10, 2, 3, 4, 10, 10, 1, 2, 3, 40])
# wvol = np.zeros(wvap.shape[0])

# print(calculate_volume_profile(wvap, wvol, 10, 3))

dataframe:pd.DataFrame = pd.DataFrame({'TRADES_average': wvap, 'TRADES_volume': wvol})
print(add_volume_profile(dataframe, volume_profile_depths=(12, 6), depth_to_bins_coeff=2))
print(add_top_of_volume_profile(dataframe, volume_profile_depths=(12, 6), depth_to_bins_coeff=2))

# hist, bins = np.histogram(wvap, bins=5, weights=wvol, density=False)
# hist1 = hist / np.sum(hist)

# print (hist1) 
# print (bins)
