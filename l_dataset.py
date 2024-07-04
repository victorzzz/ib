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
        # Extract the input data for the current window
        window = self.data.iloc[idx:(idx + self.history_len)][self.input_columns].to_numpy()
        
        # Extract the prediction data for the current window
        pred_window = self.data.iloc[idx + self.history_len:idx + self.history_len + self.pred_distance][self.pred_columns].to_numpy()
        
        # Convert the input data to a tensor
        x_tensor = torch.tensor(window, dtype=torch.float32).view(self.history_len, self.x_len)
        
        # Initialize tensors to store min and max values for each prediction column
        min_vals = torch.zeros(self.y_len, dtype=torch.float32)
        max_vals = torch.zeros(self.y_len, dtype=torch.float32)

        for i, col in enumerate(self.pred_columns):
            # Extract the prediction column data for the current prediction column
            pred_col_data = pred_window[:, i]

            # Compute the min and max values for this column
            min_vals[i] = torch.tensor(pred_col_data.min(), dtype=torch.float32)
            max_vals[i] = torch.tensor(pred_col_data.max(), dtype=torch.float32)
        
        # Concatenate min and max values to form the target output tensor
        y_tensor = torch.cat((min_vals, max_vals))
        
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
        y = np.zeros((result_rows, y_len * 2), dtype=np.float32)        
        
        # Convert relevant part of DataFrame to NumPy array for faster slicing
        data_np = data[input_columns + pred_columns].to_numpy(dtype=np.float32)
        
        for i in range(result_rows):
            x[i] = data_np[i:i + max_history_len, :x_len]
            
            # Initialize arrays to store min and max values for the current row
            min_vals = np.zeros(y_len, dtype=np.float32)
            max_vals = np.zeros(y_len, dtype=np.float32)
            
            for j, col in enumerate(pred_columns):
                # Extract the prediction sequence for the current prediction column
                pred_seq = data_np[i + max_history_len:i + max_history_len + pred_distance, pred_columns.index(col) + x_len]
                
                # Calculate min and max values for the current prediction column
                min_vals[j] = np.min(pred_seq)
                max_vals[j] = np.max(pred_seq)
            
            # Concatenate min and max values to form the target output
            y[i] = np.concatenate((min_vals, max_vals))
        
        return x, y

    @staticmethod
    def get_unique_strings(data: list[tuple[int, list[str]]]) -> list[str]:
        unique_strings = set()
        for _, strings in data:
            unique_strings.update(strings)
        return list(unique_strings)
