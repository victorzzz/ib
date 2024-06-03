import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torchmetrics
import torch.utils.data as data
import constants as cnts
import df_loader_saver as df_ls

FEATURES:int = 256
CLASSES:int = 3

class HistoricalMarketDataDataset(data.Dataset):
    def __init__(self, dfs:list[pd.DataFrame], purpose:str, shift:int = 256):
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
        x_array = np.hstack((time, price_volume_std))

        self.x = torch.tensor(x_array)
        self.y = torch.tensor(get_y_as_categories(df1), dtype=torch.long)
        self.len = self.y.shape[0] - self.shift

    def __getitem__(self, index):
        result = (create_x_row(self.x, index + self.shift), self.y[index + self.shift])
        return result

    def __len__(self):
        return self.len

class HistoricalMarketDataDataModule(L.LightningDataModule):
    def __init__(self, ticker_symbvol:str, exchange:str, batch_size:int = 256):
        super().__init__()

        self.ticker_symbvol = ticker_symbvol
        self.exchange = exchange
        self.batch_size = batch_size
        
        self.dfs = load_prepared_raw_datasets(ticker_symbvol, exchange)

    def setup(self, stage=None):
        self.train_dataset = HistoricalMarketDataDataset(self.dfs, "train")
        self.val_dataset = HistoricalMarketDataDataset(self.dfs, "val")
        self.test_dataset = HistoricalMarketDataDataset(self.dfs, "test")

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

def get_index_range_for_purpose(len:int, purpose:str) -> tuple[int, int]:
    if purpose == "train":
        return 0, int(len * 0.8)
    elif purpose == "val":
        return int(len * 0.8), int(len * 0.9)
    elif purpose == "test":
        return int(len * 0.9), len
    else:
        raise ValueError(f"Invalid purpose: {purpose}")
    
def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str) -> list[pd.DataFrame]:
    dfs:list[pd.DataFrame] = []

    for minute_multiplier in cnts.minute_multipliers.keys():
        dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
        df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
        if df is not None:
            dfs.append(df)

    return dfs

def get_y_as_categories(df:pd.DataFrame) -> np.ndarray:
    long_profit_1_5 = df["1m_long_profit_1_5"].values
    short_profit_1_5 = df["1m_short_profit_1_5"].values
    
    result_array = np.full(len(long_profit_1_5), 0, dtype=np.longlong)
    
    result_array[(long_profit_1_5 == 1) & (short_profit_1_5 == 0)] = 1
    result_array[(long_profit_1_5 == 0) & (short_profit_1_5 == 1)] = 2
    
    return result_array

def create_x_row(original_tensor: torch.Tensor, index:int) -> torch.Tensor:
    
    # Extract values for columns 0 and 1 in one go
    col0_values:torch.Tensor = original_tensor[index-127:index+1, 0].flip(dims=[0])  # This includes the current index value as the first element
    col128_values:torch.Tensor = original_tensor[index-127:index+1, 1].flip(dims=[0])  # Similarly, includes the current index value

    # Assemble the final tensor
    new_row:torch.Tensor = torch.cat((col0_values, col128_values))

    return new_row


