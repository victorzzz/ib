import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str],
        pred_distance:int):

        self.data = data
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.pred_distance = pred_distance
        
        self.history_len = max(seq[0] for seq in sequences)
        self.y_len = len(pred_columns)
        self.input_columns = TimeSeriesDataset.get_unique_strings(sequences)
        self.x_len = len(self.input_columns)
        self.result_rows = len(data) - self.history_len - pred_distance

    def __len__(self):
        return self.result_rows

    def __getitem__(self, idx):
        window = self.data.iloc[idx:(idx + self.history_len)][self.input_columns].to_numpy()
        after_window = self.data.data.iloc[idx + self.history_len + self.pred_distance][self.pred_columns].to_numpy()
        
        x_tensor = torch.tensor(window, dtype=torch.float32).view(self.history_len, self.x_len)
        y_tensor = torch.tensor(after_window, dtype=torch.float32).view(self.y_len)
        
        return x_tensor, y_tensor
    
    @staticmethod
    def to_sequences(
            data: pd.DataFrame, 
            sequences: list[tuple[int, list[str]]],
            pred_columns: list[str],
            pred_distance: int):
        
        max_history_len = max(seq[0] for seq in sequences)
        y_len = len(pred_columns)
        input_columns = TimeSeriesDataset.get_unique_strings(sequences)
        x_len = len(input_columns)
        result_rows = len(data) - max_history_len - pred_distance
        
        # Preallocate numpy arrays
        x = np.zeros((result_rows, max_history_len, x_len), dtype=np.float32)
        y = np.zeros((result_rows, y_len), dtype=np.float32)
        
        for i in range(result_rows):
            x[i] = data.iloc[i:(i + max_history_len)][input_columns].to_numpy()
            y[i] = data.iloc[i + max_history_len + pred_distance][pred_columns].to_numpy()
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def get_unique_strings(data: list[tuple[int, list[str]]]) -> list[str]:
        unique_strings = set()
        for _, strings in data:
            unique_strings.update(strings)
        return list(unique_strings)
