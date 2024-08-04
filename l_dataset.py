import logging

import numpy as np
import numpy.typing as npt

import pandas as pd
import torch
from torch.utils.data import Dataset

import l_common as lc

TIME_RANGE_DATA_FRAME_DICT = dict[int, pd.DataFrame]
TIME_RANGE_COLUMNS_LIST = list[tuple[int, list[str]]]
TIME_RANGE_COLUMNS_SEQ_LENGTH_LIST = list[tuple[int, int, list[str]]]

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df1:pd.DataFrame, 
        data:TIME_RANGE_DATA_FRAME_DICT,
        columns:TIME_RANGE_COLUMNS_SEQ_LENGTH_LIST, 
        sequences:lc.SEQUENCES_TYPE,
        pred_columns:lc.PRED_COLUMNS_TYPE):

        if any([time_range != 1 for time_range, _, _, _, _, _ in pred_columns]):
            raise ValueError("Only time range 1 is supported for prediction columns")

        self.df1 = df1
        self.data = data

        # timestamp values for each time range
        timestamp1_values = df1["1m_timestamp"].to_numpy(dtype=np.int32)
        time_stamp_values_by_time_ranges = {time_range: df["1m_timestamp"].to_numpy(dtype=np.int32) for time_range, df in data.items() if time_range != 1}
        
        # find indexes in data-frames for each time range for each timestamp in the first data-frame
        self.time_stamp_indexes = TimeSeriesDataset.find_time_stamp_indexes(time_stamp_values_by_time_ranges, timestamp1_values)
        
        self.columns = columns
        self.sequences = sequences
        self.pred_columns = pred_columns
        
        self.max_history1_len = max(seq_len for time_range, seq_len, _ in columns if time_range == 1)
        self.max_pred1_distance = max(pred_distance for _, pred_distance, _, _, _, _ in pred_columns)
        self.df1_len = len(df1)
        
        self.result_rows = self.df1_len - self.max_history1_len - self.max_pred1_distance - 1
        
    def __len__(self):
        return self.result_rows

    def __getitem__(self, idx):
        x_result_tensor:torch.Tensor | None = None
        y_result_tensor:torch.Tensor | None = None
        
        i1 = idx + self.max_history1_len
        
        for time_range, seq_len, columns in self.columns:            
            if time_range == 1:
                df = self.df1
                i = i1
            else:
                df = self.data[time_range]
                
                # find the closest (greate or equal) index in the data frame
                i = self.time_stamp_indexes[time_range]
            
            # Extract the input data for the current window
            window = df.iloc[(i-seq_len):i][columns].to_numpy(dtype=np.float32).reshape(-1)
                        
            # Convert the input data to a tensor
            x_tensor = torch.tensor(window, dtype=torch.float32)
            
            if x_result_tensor is None:
                x_result_tensor = x_tensor
            else:
                x_result_tensor = torch.cat((x_result_tensor, x_tensor))
            
        for time_range, pred_distance, column, pred_aggregations, pred_transformations, retio_multiplier in self.pred_columns:

            last_observed_prediction_value = self.df1.iloc[i1-1][column].to_numpy(dtype=np.float32)
            
            # Extract the prediction data for the current window
            pred_window =self.df1.iloc[i1:i1 + pred_distance][column].to_numpy(dtype=np.float32)
            
            for pred_aggregation in pred_aggregations:
                if pred_aggregation == lc.PRED_MIN:
                    pred_value = np.min(pred_window)
                elif pred_aggregation == lc.PRED_MAX:
                    pred_value = np.max(pred_window)
                elif pred_aggregation == lc.PRED_AVG:
                    pred_value = np.mean(pred_window)
                elif pred_aggregation == lc.PRED_LAST:
                    pred_value = pred_window[-1]
                elif pred_aggregation == lc.PRED_FIRST:
                    pred_value = pred_window[0]
                elif pred_aggregation == lc.PRED_LAST_OBSERVED:
                    pred_value = last_observed_prediction_value
                else:
                    raise ValueError(f"Unsupported aggregation: {pred_aggregation}")
                
                for pred_transformation in pred_transformations:
                    if pred_transformation == lc.PRED_TRANSFORM_NONE:
                        result_pred_value = pred_value
                    elif pred_transformation == lc.PRED_TRANSFORM_DIFF:
                        result_pred_value = pred_value - last_observed_prediction_value
                    elif pred_transformation == lc.PRED_TRANSFORM_RATIO:
                        result_pred_value = (pred_value / last_observed_prediction_value - 1.0) * retio_multiplier
                        
                    y_tensor = torch.tensor(result_pred_value, dtype=torch.float32)
                    
                    if y_result_tensor is None:
                        y_result_tensor = y_tensor
                    else:
                        y_result_tensor = torch.cat((y_result_tensor, y_tensor))

        return x_result_tensor, y_result_tensor
    
    """
    @staticmethod
    def to_sequences(
            data: pd.DataFrame, 
            sequences: list[tuple[int, list[str]]],
            pred_columns: list[str],
            pred_distance: int) -> tuple[np.ndarray, np.ndarray]:
        
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
        """

    """
    @staticmethod
    def get_unique_strings(data: list[tuple[int, list[str]]]) -> list[str]:
        unique_strings = set()
        for _, strings in data:
            unique_strings.update(strings)
        return list(unique_strings)
    """
    
    """
    @staticmethod
    # each tuple: (candle sticks time range, sequence length, data_types, ema_periods, data_columns)
    def get_used_columns(seq:list[tuple[int, int, list[str], list[int], list[str]]]) -> list[tuple[int, list[str]]]:
        
        # key: time range, value: set of columns
        accumulator:dict[int, set[str]] = {}
        
        for time_range, seq_len, data_types, ema_periods, columns in seq:
            if time_range not in accumulator:
                accumulator[time_range] = set()
            accumulator[time_range].update(columns)
        
        return [(time_range, list(columns)) for time_range, columns in accumulator.items()]
    """
    
    @staticmethod
    def get_columns_from_pred_columns(pred_columns:lc.PRED_COLUMNS_TYPE) -> TIME_RANGE_COLUMNS_LIST:
        
        # key: time range, value: set of columns
        accumulator:dict[int, set[str]] = {}
        
        for time_range, _, column, _, _, _ in pred_columns:
            if time_range not in accumulator:
                accumulator[time_range] = set()
            accumulator[time_range].update([column])
            
        return [(time_range, list(columns)) for time_range, columns in accumulator.items()]

    @staticmethod    
    def find_index_for_timestamp(array:npt.NDArray[np.int32], timestamp:np.int32) -> np.int32:
        index = np.searchsorted(array, timestamp, side='right') - 1
        if index >= 0 and array[index] == timestamp:
            return index
        else:
            raise ValueError(f"Corresponding timestamp for {timestamp} not found in array.")
    
    @staticmethod 
    def find_time_stamp_indexes(time_stamp_values:dict[int, npt.NDArray[np.int32]], time_stamp_value1:npt.NDArray[np.int32]) -> dict[int, npt.NDArray[np.int32]]:
        time_stamp_indexes = {}
        for time_range, ts_values in time_stamp_values.items():
            time_stamp_indexes[time_range] = np.array([TimeSeriesDataset.find_index_for_timestamp(time_stamp_value1, time_stamp) for time_stamp in ts_values], dtype=np.int32)
        return time_stamp_indexes
    
    @staticmethod
    def merge_columns_for_time_ranges(columns_sets:list[TIME_RANGE_COLUMNS_LIST]) -> TIME_RANGE_COLUMNS_LIST:
        
        # key: time range, value: set of columns
        accumulator:dict[int, set[str]] = {}
        
        for columns_set in columns_sets:
            for time_range, columns in columns_set:
                if time_range not in accumulator:
                    accumulator[time_range] = set()
                accumulator[time_range].update(columns)
            
        return [(time_range, list(columns)) for time_range, columns in accumulator.items()]
    
