import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data
import constants as cnts
import df_loader_saver as df_ls
import logging
import torch

HISTORY_SIZE:int = 1024
FEATURES:int = HISTORY_SIZE * 2 + 4
CLASSES:int = 3


class HistoricalMarketDataDataset(data.Dataset):
    def __init__(self, dfs:list[pd.DataFrame], purpose:str, shift:int = HISTORY_SIZE):
        super().__init__()

        self.shift = shift

        df1 = dfs[0]
        origin_len = df1.shape[0]
        
        start, end = get_index_range_for_purpose(origin_len, purpose)
        df1 = df1[start:end]

        price_volume = df1[["1m_TRADES_average", "1m_TRADES_volume"]].values
        
        sc_x = StandardScaler()
        price_volume_std = sc_x.fit_transform(price_volume)

        time = df1[["1m_normalized_day_of_week", "1m_normalized_week", "1m_normalized_day_of_year", "1m_normalized_trading_time"]].values
        y = get_y_as_categories(df1).reshape(-1, 1)
        
        self.array = np.hstack((price_volume_std, time, y))
        self.len = self.array.shape[0] - self.shift

    def __getitems__(self, indexes:list[int]):
        np_indexes = np.array(indexes)
        shifted_indexes = np_indexes + self.shift
        
        result = create_rows(self.array, shifted_indexes)
        return result
        
    def __len__(self):
        return self.len
    
class HistoricalMarketDataDataModule(L.LightningDataModule):
    def __init__(self, ticker_symbvol:str, exchange:str, batch_size:int = 512):
        super().__init__()

        self.ticker_symbvol = ticker_symbvol
        self.exchange = exchange
        self.batch_size = batch_size

    def setup(self, stage=None):
        dfs = load_prepared_raw_datasets(self.ticker_symbvol, self.exchange)
        
        self.train_dataset = HistoricalMarketDataDataset(dfs, "train")
        self.val_dataset = HistoricalMarketDataDataset(dfs, "val")
        self.test_dataset = HistoricalMarketDataDataset(dfs, "test")
        
        logging.info(f"Datasets were loaded train_dataset: {len(self.train_dataset)}")
        
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=collate_fn, 
            shuffle=True,
            persistent_workers=False)
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            collate_fn=collate_fn, 
            shuffle=False,
            persistent_workers=False)
    
    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            collate_fn=collate_fn, 
            shuffle=False,
            persistent_workers=False)

def collate_fn(batch):
    features = torch.tensor(batch[:, :-1]).float()
    labels = torch.tensor(batch[:, -1]).long()
    
    return features, labels

def get_index_range_for_purpose(len:int, purpose:str) -> tuple[int, int]:
    if purpose == "train":
        return int(len * 0.05), int(len * 0.8)
    elif purpose == "val":
        return int(len * 0.8), int(len * 0.9)
    elif purpose == "test":
        return int(len * 0.9), len
    else:
        raise ValueError(f"Invalid purpose: {purpose}")
    
def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str) -> list[pd.DataFrame]:
    logging.info(f"Loading prepared raw datasets for {ticker_symbvol} on {exchange} ...")
    
    dfs:list[pd.DataFrame] = []

    for minute_multiplier in cnts.minute_multipliers.keys():
        dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
        df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
        if df is not None:
            dfs.append(df)

    return dfs

def get_y_as_categories(df:pd.DataFrame) -> np.ndarray:
    long_profit_1_5 = df["1m_long_profit_0_9"].values
    short_profit_1_5 = df["1m_short_profit_0_9"].values
    
    result_array = np.full(len(long_profit_1_5), 0, dtype=np.float32)
    
    result_array[(long_profit_1_5 == 1) & (short_profit_1_5 == 0)] = 1
    result_array[(long_profit_1_5 == 0) & (short_profit_1_5 == 1)] = 2
    
    return result_array

"""
def create_x_row(original: np.ndarray, index:int) -> np.ndarray:
    
    # Extract values for columns 0 and 1 in one go
    col0_values = np.flip(original[index-127:index+1, 0])  # This includes the current index value as the first element
    col128_values = np.flip(original[index-127:index+1, 1])  # Similarly, includes the current index value

    # Extract the last four columns (2, 3, 4, 5) from the current index
    last_four_values = original[index, 2:6]

    # Assemble the final array
    new_row = np.concatenate((col0_values, col128_values, last_four_values))

    return new_row
"""

def create_rows(original: np.ndarray, indexes:np.ndarray) -> np.ndarray:
    
    # Determine number of indices to process
    num_indices = len(indexes)

    # Pre-allocate results array
    results = np.empty((num_indices, HISTORY_SIZE*2 + 4 + 1), dtype=original.dtype)

    # Fetch the last four columns for all specified indices at once
    last_four_values = original[indexes, 2:6]

    # Process columns 0 and 1 with flipping
    for i, index in enumerate(indexes):
        col0_values = np.flip(original[index-HISTORY_SIZE+1:index+1, 0])
        col128_values = np.flip(original[index-HISTORY_SIZE+1:index+1, 1])

        results[i, :HISTORY_SIZE] = col0_values
        results[i, HISTORY_SIZE:HISTORY_SIZE*2] = col128_values
    
    results[:, HISTORY_SIZE*2:HISTORY_SIZE*2 + 4] = last_four_values[i]            
    results[:, -1] = original[indexes, -1]        

    return results
