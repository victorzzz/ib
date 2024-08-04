import logging

import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt

import l_dataset as l_ds
import df_loader_saver as df_ls

import constants as cnts
import l_common as lc
import df_tech_indicator_utils as df_ti_utils

import ib_logging as ib_log

class StockPriceDataModule(L.LightningDataModule):
    def __init__(
        self, 
        ticker_symbvol:str, 
        exchange:str,
        
        time_ranges:list[int],
        
        sequences:lc.SEQUENCES_TYPE,
        pred_columns:lc.PRED_COLUMNS_TYPE,
        log_columns:lc.LOG_COLUMNS_TYPE,
        scaling_column_groups:lc.SCALING_COLUMN_GROUPS_TYPE,

        # user_tensor_dataset:bool, 
        
        batch_size,
        tail:float,
        train_part:float=0.8,
        keep_loaded_data:bool=False,
        keep_scaled_data:bool=False,
        keep_validation_dataset:bool=False,
        keep_train_dataset:bool=False):
        
        super(StockPriceDataModule, self).__init__()
        
        logging.info(f"StockPriceDataModule.__init__ : {ticker_symbvol} on {exchange} ...")
        
        # lumbda functions for creating scalers
        self.scaler_factory = lambda: MinMaxScaler((-1, 1))
        
        self.ticker_symbvol = ticker_symbvol
        self.exchange = exchange
        
        self.time_ranges = time_ranges
        self.sequences = sequences
        self.pred_columns = pred_columns
        
        self.log_columns = log_columns
        
        self.scaling_column_groups = scaling_column_groups
        # self.user_tensor_dataset = user_tensor_dataset
        self.batch_size = batch_size
        self.tail = tail
        self.train_part = train_part
        self.keep_loaded_data = keep_loaded_data
        self.keep_scaled_data = keep_scaled_data
        self.keep_validation_dataset = keep_validation_dataset
        self.keep_train_dataset = keep_train_dataset

        self.scalers:dict[str, StandardScaler | RobustScaler | MinMaxScaler] = {
            self.scaler_key(fitting_time_range, fitting_column): self.scaler_factory()
                for (fitting_time_range, fitting_column), _ in scaling_column_groups
            }

        self.train_dataset : Dataset | None = None
        self.val_dataset : Dataset | None = None
        
        self.data_was_prepared = False

    def prepare_data(self) -> None:
        
        if self.data_was_prepared:
            logging.info("Data was already prepared")
            return
        
        logging.info(f"StockPriceDataModule.prepare_data : {self.ticker_symbvol} on {self.exchange} for time ranges {self.time_ranges} ...")
        
        # Load prepared raw datasets
        data_frames:dict[int, pd.DataFrame] = StockPriceDataModule.load_prepared_raw_datasets(self.ticker_symbvol, self.exchange, self.time_ranges)
        
        # Add log columns
        self.add_log_columns(data_frames, self.log_columns)
        
        # Add augmented columns
        used_columns, used_columns_seq_length = self.add_augmented_columns(data_frames, self.sequences)
        
        pred_columns:l_ds.TIME_RANGE_COLUMNS_LIST = l_ds.TimeSeriesDataset.get_columns_from_pred_columns(self.pred_columns)
        used_columns = l_ds.TimeSeriesDataset.merge_columns_for_time_ranges([used_columns, pred_columns])
        
        # Get the only used colums from leaded datasets
        data_frames = self.data_frames_with_columns(data_frames, used_columns)
        
        # Set appropriate indexes
        data_frames = self.set_indexes(data_frames)
        
        if self.keep_loaded_data:
            self.loaded_fs = self.copy_dataframes(data_frames)
        
        logging.info(f"Used columns: {used_columns}")
        
        df1 = data_frames[1]
        
        original_df1_length:int = len(data_frames[1])
        train_val_border:int = int(original_df1_length * self.train_part)
        
        logging.info(f"Original dataset length: {original_df1_length}, train_val_border: {train_val_border}")
        
        training_df1_orig:pd.DataFrame = df1[:train_val_border].reset_index(drop=True)
        val_df1_orig:pd.DataFrame = df1[train_val_border:].reset_index(drop=True)
    
        if self.keep_loaded_data:
            self.training_df1_orig = training_df1_orig.copy()
            self.val_df1_orig = val_df1_orig.copy()
    
        logging.info("Fitting scalers ...")
        self.fit(training_df1_orig)
        
        logging.info("Transforming training data ...")
        self.transform(training_df1_orig, data_frames)
        
        logging.info("Transforming validation data ...")
        self.transform(val_df1_orig, data_frames)
        
        if self.keep_scaled_data:
            self.tdf1 = training_df1_orig.copy()
            self.vdf1 = val_df1_orig.copy()
            
        training_df_description = training_df1_orig.describe().T
        logging.info(f"Scaled training df description:\n{training_df_description.sort_values(by=['max'], ascending=False)}")

        val_df_description = val_df1_orig.describe().T
        logging.info(f"Scaled validation df description:\n {val_df_description.sort_values(by=['max'], ascending=False)}")
        
        self.check_scaled_df_description(training_df_description)
        self.check_scaled_df_description(val_df_description)
        
        for time_range, df in data_frames.items():
            if time_range == 1:
                continue
            
            df_description = df.describe().T
            logging.info(f"Scaled df {time_range}:\n {df_description.sort_values(by=['max'], ascending=False)}")
            self.check_scaled_df_description(df_description)
            
        logging.info("Creating datasets ...")
        """
        if self.user_tensor_dataset:
            logging.info("Training tensor dataset ...")
            train_src, train_y = l_ds.TimeSeriesDataset.to_sequences(training_df, self.sequences, self.pred_columns, self.pred_distance)
            
            if self.keep_train_dataset:
                self.train_scr = train_src
                self.train_y = train_y
            
            x = torch.tensor(train_src, dtype=torch.float32)
            y = torch.tensor(train_y, dtype=torch.float32)
            
            self.train_dataset = TensorDataset(x, y)
            
            logging.info("Validation tensor dataset ...")
            val_scr, val_y = l_ds.TimeSeriesDataset.to_sequences(val_df, self.sequences, self.pred_columns, self.pred_distance)
            
            if self.keep_validation_dataset:
                self.val_scr = val_scr
                self.val_y = val_y
            
            x = torch.tensor(val_scr, dtype=torch.float32)
            y = torch.tensor(val_y, dtype=torch.float32)   
                     
            self.val_dataset = TensorDataset(x, y)            
        else:
        """

        logging.info("Training dataset ...")
        # self.train_dataset = l_ds.TimeSeriesDataset(training_df, self.sequences, self.pred_columns, self.pred_distance)
        
        logging.info("Validation dataset ...")
        # self.val_dataset = l_ds.TimeSeriesDataset(val_df, self.sequences, self.pred_columns, self.pred_distance)
        
        self.data_was_prepared = True
    
    def setup(self, stage=None):
        logging.info(f"StockPriceDataModule.setup : {stage}")

    def train_dataloader(self):
        if isinstance(self.train_dataset, Dataset):
            logging.info(f"StockPriceDataModule.train_dataloader ...")
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            raise ValueError("train_dataset is not initialized")

    def val_dataloader(self):
        if isinstance(self.val_dataset, Dataset):
            logging.info(f"StockPriceDataModule.val_dataloader ...")
            return DataLoader(self.val_dataset, batch_size=self.batch_size)
        else:
            raise ValueError("val_dataset is not initialized")
    
    def train_dataloader_not_shuffled(self):
        if isinstance(self.train_dataset, Dataset):
            logging.info(f"StockPriceDataModule.train_dataloader_not_shuffled ...")
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError("train_dataset is not initialized")
    
    def predict_dataloader(self):
        return self.val_dataloader()
    
    def on_exception(self, exception: BaseException) -> None:
        logging.error(f"StockPriceDataModule.on_exception : {exception}")
    
    def teardown(self, stage: str) -> None:
        logging.info(f"StockPriceDataModule.teardown : {stage}")
    
    ######################
    def fit(self, df_1:pd.DataFrame) -> None:
        for (fitting_time_range, fitting_column), _ in self.scaling_column_groups:
                if fitting_time_range != 1:
                    raise ValueError("Fitting time range should be 1")
                
                scalers_dict_key = self.scaler_key(fitting_time_range, fitting_column)
                scaler = self.scalers[scalers_dict_key]
                self.fit_on_column(scaler, df_1, fitting_column)
                
        self.was_fit = True        
        
    """
    ######################
    def fit_transform(self, df_1:pd.DataFrame, dfs:dict[int, pd.DataFrame]) -> None:
        for (fitting_time_range, fitting_column), scaling_columns in self.scaling_column_groups:
                scalers_dict_key = self.scaler_key(fitting_time_range, fitting_column)
                if fitting_time_range == 1:
                   fitting_df = df_1
                else:
                    fitting_df = dfs[fitting_time_range] 
                scaler = self.scalers[scalers_dict_key]
                self.fit_transform_column(scaler, fitting_df, fitting_column)
                
                for scaling_time_range, columns in scaling_columns:
                    if scaling_time_range == 1:
                        scaling_df = df_1
                    else:
                        scaling_df = dfs[scaling_time_range]

                    self.transform_columns(scaler, scaling_df, columns)
        
        self.was_fit = True
    """
    
    ######################
    def transform(self, df_1:pd.DataFrame, dfs:dict[int, pd.DataFrame]) -> None:
        for (fitting_time_range, fitting_column), scaling_columns in self.scaling_column_groups:
            scalers_dict_key = self.scaler_key(fitting_time_range, fitting_column)
            scaler = self.scalers[scalers_dict_key]

            for scaling_time_range, columns in scaling_columns:
                if scaling_time_range == 1:
                    scaling_df = df_1
                    columns = [fitting_column] + columns
                else:
                    scaling_df = dfs[scaling_time_range]

                self.transform_columns(scaler, scaling_df, columns)

    #####################    
    def inverse_transform_predictions(self, predictions:np.ndarray, fiting_column:str, number_of_columns:int) -> np.ndarray:
        scaler:StandardScaler | RobustScaler | MinMaxScaler = self.scalers[fiting_column]
        scaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        if isinstance(scaled_predictions, np.ndarray):
            return scaled_predictions.reshape(-1, number_of_columns)
        else:
            raise ValueError("Predictions should be numpy array")

    """
    def get_df(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.keep_loaded_data:
            return self.df.copy(), self.training_df_orig.copy(), self.val_df_orig.copy()
        else:
            raise ValueError("Data was not kept")

    def get_scaled_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.keep_scaled_data:
            return self.tdf.copy(), self.vdf.copy()
        else:
            raise ValueError("Scaled data was not kept")

    def get_val_src_y(self) -> tuple[np.ndarray, np.ndarray]:
        if self.keep_validation_dataset and self.user_tensor_dataset:
            return self.val_scr, self.val_y
        else:
            raise ValueError("val_scr and val_y are not initialized")

    def get_train_src_y(self) -> tuple[np.ndarray, np.ndarray]:
        if self.keep_train_dataset and self.user_tensor_dataset:
            return self.train_scr, self.train_y
        else:
            raise ValueError("val_scr and val_y are not initialized")
    """
    
    @staticmethod
    def check_scaled_df_description(scaled_df_description:pd.DataFrame):
        desc_without_timestamp_df = scaled_df_description[~(scaled_df_description.index.str.endswith('_timestamp'))]
        
        desc_without_timestamp_df_min_5 = desc_without_timestamp_df[desc_without_timestamp_df["min"] < -5.0]
        if len(desc_without_timestamp_df_min_5) > 0:
            logging.info(f"Min < -5.0:\n {desc_without_timestamp_df_min_5}")
            logging.critical("Scaled data is out of range [-5, 5] for columns: {min_errors}")
            raise ValueError("Scaled data is out of range [-5, 5] for columns: {min_errors}")
        
        desc_without_timestamp_df_max_5 = desc_without_timestamp_df[desc_without_timestamp_df["max"] > 5.0]
        if len(desc_without_timestamp_df_max_5) > 0:
            logging.info(f"Max > 5.0:\n {desc_without_timestamp_df_max_5}")
            logging.critical("Scaled data is out of range [-5, 5] for columns: {max_errors}")
            raise ValueError("Scaled data is out of range [-5, 5] for columns: {max_errors}")
        
    @staticmethod
    def fit_on_column(scaler:StandardScaler | RobustScaler | MinMaxScaler, df:pd.DataFrame, column:str):
        values_to_fit = df[[column]].to_numpy()
        scaler.fit(values_to_fit)

    @staticmethod
    def transform_columns(scaler:StandardScaler | RobustScaler | MinMaxScaler, 
                          df:pd.DataFrame,
                          columns:list[str]) -> None:
        for column in columns:
            values_to_transform = df[[column]].to_numpy()
            transformed_vale = scaler.transform(values_to_transform)
            if isinstance(transformed_vale, np.ndarray):
                df[[column]] = transformed_vale
    
    @staticmethod
    def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str, time_ranges:list[int]) -> l_ds.TIME_RANGE_DATA_FRAME_DICT:
        logging.info(f"Loading prepared raw datasets for {ticker_symbvol} on {exchange} ...")
        
        dfs:l_ds.TIME_RANGE_DATA_FRAME_DICT = {}

        for minute_multiplier in cnts.minute_multipliers.keys():
            int_minute_multiplier = int(minute_multiplier)
            if int_minute_multiplier not in time_ranges:
                continue
            
            dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
            df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
            if df is not None:
                dfs[int_minute_multiplier] = df
                
        return dfs
    
    @staticmethod
    def add_log_columns(data_frames:l_ds.TIME_RANGE_DATA_FRAME_DICT, log_columns:list[tuple[int, str]]) -> None:
        for time_range, column in log_columns:
            df = data_frames[time_range]
            df[f'{column}_LOG'] = np.log(df[column] + 0.0001)
    
    @staticmethod
    def data_frames_with_columns(data_frames:l_ds.TIME_RANGE_DATA_FRAME_DICT, used_columns:list[tuple[int, list[str]]]) -> l_ds.TIME_RANGE_DATA_FRAME_DICT:
        result = {}
        for time_range, columns in used_columns:
            timestamp_column = f"{time_range}m_timestamp"
            columns = [timestamp_column] + columns
            df = data_frames[time_range]
            result[time_range] = df[columns].copy()
        
        return result
    
    @staticmethod
    def set_indexes(data_frames:l_ds.TIME_RANGE_DATA_FRAME_DICT) -> l_ds.TIME_RANGE_DATA_FRAME_DICT:
        for time_range, df in data_frames.items():

            if time_range != 1:
                df.set_index(f"{time_range}m_timestamp", drop=False, inplace=True)
            else:
                df.sort_values(by="1m_timestamp", ascending=True, inplace=True)
                df.reset_index(drop=True, inplace=True)
        
        return data_frames    
    
    # returns used columns for each time range
    @staticmethod
    def add_augmented_columns(data_frames:l_ds.TIME_RANGE_DATA_FRAME_DICT,
                              sequences:lc.SEQUENCES_TYPE) -> tuple[l_ds.TIME_RANGE_COLUMNS_LIST, l_ds.TIME_RANGE_COLUMNS_SEQ_LENGTH_LIST]:
        result_columns = []
        result_columns_seq_length = []
        
        for time_range, sequence_length, data_types, ema_periods, data_columns in sequences:
            
            if lc.DATA_CATEGORY in data_types:
                continue
            
            df = data_frames[time_range]
            
            add_ema_colums_to_df:bool = lc.DATA_EMA in data_types
            add_ema_dif_columns_to_df:bool = lc.DATA_EMA_DIFF in data_types
            add_ema_retio_columns_to_df:bool = lc.DATA_EMA_RATIO in data_types
            
            new_df, ema_columns, ema_dif_columns, ema_ratio_columns = df_ti_utils.add_ema(
                df, 
                data_columns, 
                ema_periods, 
                add_ema_colums_to_df, 
                add_ema_dif_columns_to_df, 
                add_ema_retio_columns_to_df)
            
            data_frames[time_range] = new_df.copy()

            data_columns += ema_columns       
            data_columns += ema_dif_columns
            data_columns += ema_ratio_columns
            
            data_columns = list(set(data_columns))
            
            result_columns.append((time_range, data_columns))
            
            use_data_column:bool = lc.DATA_VALUE in data_types
            used_columns:list[str] = [] if not use_data_column else data_columns
            used_columns = used_columns if not add_ema_colums_to_df else used_columns + ema_columns
            used_columns = used_columns if not add_ema_dif_columns_to_df else used_columns + ema_dif_columns
            used_columns = used_columns if not add_ema_retio_columns_to_df else used_columns + ema_ratio_columns
            
            result_columns_seq_length.append((time_range, sequence_length, used_columns))
            
        return (result_columns, result_columns_seq_length)
    
    @staticmethod
    def scaler_key(fitting_time_range:int, fitting_column:str) -> str:
        return f"{fitting_time_range}_{fitting_column}"
    
    @staticmethod
    def copy_dataframes(dfs:l_ds.TIME_RANGE_DATA_FRAME_DICT, exclude_time_range:int | None = None) -> l_ds.TIME_RANGE_DATA_FRAME_DICT:
        result = {}
        for time_range, df in dfs.items():
            if (exclude_time_range ) and (time_range != exclude_time_range):
                result[time_range] = df.copy()
        
        return result
    
if __name__ == "__main__":
    
    ib_log.configure_logging("training")

    logging.info(f"Starting {__file__} ...")

    torch.set_float32_matmul_precision('high')    
    
    # Create data module
    data_module = StockPriceDataModule (
        ticker_symbvol="RY", 
        exchange="TSE",
        time_ranges=[1, 3, 10, 30],
        tail=lc.dataset_tail,
        sequences=lc.sequences,
        pred_columns=lc.pred_columns,
        log_columns=lc.log_columns,
        scaling_column_groups=lc.scaling_column_groups,
        # pred_distance=lc.prediction_distance,
        # user_tensor_dataset=True,
        batch_size=lc.batch_size_param,
        keep_loaded_data=False,
        keep_scaled_data=False,
        keep_validation_dataset=False
    )
    
    data_module.prepare_data()