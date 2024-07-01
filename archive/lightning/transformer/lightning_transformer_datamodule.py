import df_loader_saver as df_ls
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import pandas as pd

import constants as cnts

import logging
import ib_logging as ib_log

DF_START_THRESHOLD:float = 0.01

class TimeSeriesTransformerDataset(Dataset):
    def __init__(self, 
                 df:pd.DataFrame,
                 sequences:list[tuple[int, list[str]]], 
                 pred_columns:list[str], 
                 pred_len:int=8,
                 transformer_target_shift:int=1, 
                 step:int=2):
        super().__init__()
        
        """
        df: DataFrame
        sequences: List of tuples [(history_len, [column names]), ...]
        pred_columns: List of column names for the predicted values
        pred_len: Number of future steps to predict
        step: Step size for selecting future values
        """
        self.df = df
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.pred_len = pred_len
        self.transformer_target_shift = transformer_target_shift,
        self.step = step
        self.max_history_len = max(seq[0] for seq in sequences)
        
    def __len__(self):
        return len(self.df) - self.max_history_len - (self.pred_len * self.step)

    def __getitem__(self, idx):
        src_sequences = []
        tgt_sequences = []
        
        for history_len, columns in self.sequences:
            # Prepare source sequences
            src = self.df.iloc[idx:idx+history_len][columns].values
            src_sequences.append(src)
            
            # Prepare target sequences
            tgt = self.df.iloc[idx+self.transformer_target_shift:idx+history_len+self.transformer_target_shift][columns].values
            tgt_sequences.append(tgt)
        
        # Concatenate all sequences
        src = np.concatenate(src_sequences, axis=1)
        tgt = np.concatenate(tgt_sequences, axis=1)
        
        # Select future values with a step for prediction
        y_indices = list(range(idx+self.max_history_len, idx+self.max_history_len+(self.pred_len*self.step), self.step))
        y = self.df.iloc[y_indices][self.pred_columns].values  # Use pred_columns for y
        
        return torch.tensor(src, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class StockTransformerDataModule(L.LightningDataModule):
    def __init__(self, 
                 ticker_symbvol:str, 
                 exchange:str, 
                 sequences:list[tuple[int, list[str]]], 
                 pred_columns:list[str], 
                 batch_size=32, 
                 pred_len=8, 
                 step=2,
                 train_part:float=0.8):
        super().__init__()
        
        self.ticker_symbvol = ticker_symbvol
        self.exchange = exchange
        self.batch_size = batch_size
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.pred_len = pred_len
        self.step = step
        self.train_part = train_part

    def setup(self, stage=None):
        dfs = load_prepared_raw_datasets(self.ticker_symbvol, self.exchange)
        
        self.df0 = dfs[0]
        original_df_length:int = len(self.df0)
        
        train_val_border:int = int(original_df_length * self.train_part)
        
        training_df:pd.DataFrame = self.df0[int(original_df_length * DF_START_THRESHOLD):train_val_border].copy()
        val_df:pd.DataFrame = self.df0[train_val_border:].copy()        

        self.train_dataset = TimeSeriesTransformerDataset(training_df, self.sequences, self.pred_columns, self.pred_len, self.step)
        self.val_dataset = TimeSeriesTransformerDataset(val_df, self.sequences, self.pred_columns, self.pred_len, self.step)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str) -> list[pd.DataFrame]:
    logging.info(f"Loading prepared raw datasets for {ticker_symbvol} on {exchange} ...")
    
    dfs:list[pd.DataFrame] = []

    for minute_multiplier in cnts.minute_multipliers.keys():
        dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
        df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
        if df is not None:
            dfs.append(df)

    return dfs