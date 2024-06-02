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

class HistoricalMarketDataDataset(data.Dataset):
    def __init__(self, ticker_symbvol:str, exchange:str):
        super().__init__()

        self.dfs = load_prepared_raw_datasets(ticker_symbvol, exchange)

        df1 = self.dfs[1]
        price_volume = df1[["1m_TRADES_average", "1m_TRADES_volume"]].values
        
        sc_x = StandardScaler()
        price_volume_std = sc_x.fit_transform(price_volume)

        time = df1[["1m_normalized_day_of_week", "1m_normalized_week", "1m_normalized_day_of_year", "1m_normalized_trading_time"]].values
        x_array = np.hstack((time, price_volume_std))

        self.x = torch.tensor(x_array, dtype=torch.float16)
        self.y = torch.tensor(get_y_as_categories(df1), dtype=torch.int8)

    def __getitem__(self, index):
        return self.x[index + 256], self.y[index + 256]

    def __len__(self):
        return self.y.shape[0] - 256
    
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
    
    result_array = np.full(len(long_profit_1_5), 0, dtype=np.int8)
    
    result_array[(long_profit_1_5 == 1) & (short_profit_1_5 == 0)] = 1
    result_array[(long_profit_1_5 == 0) & (short_profit_1_5 == 1)] = 2
    
    return result_array

def create_x_row(original_tensor: torch.Tensor, index:int) -> torch.Tensor:
    
    # Extract values for columns 0 and 1 in one go
    col0_values:torch.Tensor = original_tensor[index-127:index+1, 0].flip(dims=[0])  # This includes the current index value as the first element
    col128_values:torch.Tensor = original_tensor[index-127:index+1, 1].flip(dims=[0])  # Similarly, includes the current index value

    # Extract the last four columns (2, 3, 4, 5) from the current index
    last_four_values:torch.Tensor = original_tensor[index, 2:6]

    # Assemble the final tensor
    new_row:torch.Tensor = torch.cat((col0_values, col128_values, last_four_values))

    return new_row


