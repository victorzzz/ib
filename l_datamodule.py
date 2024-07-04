import logging

import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import l_dataset as l_ds
import df_loader_saver as df_ls

import constants as cnts

class StockPriceDataModule(L.LightningDataModule):
    def __init__(
        self, 
        ticker_symbvol:str, 
        exchange:str,  
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str],
        scaling_column_groups:dict[str, tuple[list[str], bool]], # {fiting_column: ([scaling1, column2, ...], Log_before_scaling)}
        pred_distance:int,
        user_tensor_dataset:bool, 
        batch_size,
        tail:float=0.075,
        train_part:float=0.8,
        keep_loaded_data:bool=False,
        keep_scaled_data:bool=False,
        keep_validation_dataset:bool=False,
        keep_train_dataset:bool=False):
        
        super(StockPriceDataModule, self).__init__()
        
        logging.info(f"StockPriceDataModule.__init__ : {ticker_symbvol} on {exchange} ...")
        
        self.ticker_symbvol = ticker_symbvol
        self.exchange = exchange
        
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.scaling_column_groups = scaling_column_groups
        self.pred_distance = pred_distance
        self.user_tensor_dataset = user_tensor_dataset
        self.batch_size = batch_size
        self.tail = tail
        self.train_part = train_part
        self.keep_loaded_data = keep_loaded_data
        self.keep_scaled_data = keep_scaled_data
        self.keep_validation_dataset = keep_validation_dataset
        self.keep_train_dataset = keep_train_dataset
        
        self.scalers:dict[str, StandardScaler] = {fitting_column: StandardScaler() for fitting_column in scaling_column_groups}
        
        self.train_dataset : Dataset | None = None
        self.val_dataset : Dataset | None = None
        
        self.data_was_prepared = False

    def prepare_data(self) -> None:
        
        if self.data_was_prepared:
            logging.info("Data was already prepared")
            return
        
        logging.info(f"StockPriceDataModule.prepare_data : {self.ticker_symbvol} on {self.exchange} ...")
        
        data_frames:list[pd.DataFrame] = StockPriceDataModule.load_prepared_raw_datasets(self.ticker_symbvol, self.exchange)
        df0:pd.DataFrame = data_frames[0].tail(round(len(data_frames[0]) * self.tail)).reset_index(drop=True)
        
        used_columns = l_ds.TimeSeriesDataset.get_unique_strings(self.sequences)
        used_columns.extend(self.pred_columns)
        used_columns = list(set(used_columns))
        
        df0 = df0[used_columns].copy()
        
        if self.keep_loaded_data:
            self.df = df0.copy()
        
        logging.info(f"Used columns: {df0.columns}")
        
        original_df_length:int = len(df0)
        train_val_border:int = int(original_df_length * self.train_part)
        
        logging.info(f"Original dataset length: {original_df_length}, train_val_border: {train_val_border}")
        
        training_df_orig:pd.DataFrame = df0[:train_val_border].reset_index(drop=True)
        val_df_orig:pd.DataFrame = df0[train_val_border:].reset_index(drop=True)
    
        if self.keep_loaded_data:
            self.training_df_orig = training_df_orig
            self.val_df_orig = val_df_orig
    
        logging.info("Fitting scalers ...")
        training_df = self.fit_transform(training_df_orig.copy())
        
        logging.info("Transforming data ...")
        val_df = self.transform(val_df_orig.copy())
        
        if self.keep_scaled_data:
            self.tdf = training_df
            self.vdf = val_df
                
        logging.info("Creating datasets ...")
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
            logging.info("Training dataset ...")
            self.train_dataset = l_ds.TimeSeriesDataset(training_df, self.sequences, self.pred_columns, self.pred_distance)
            
            logging.info("Validation dataset ...")
            self.val_dataset = l_ds.TimeSeriesDataset(val_df, self.sequences, self.pred_columns, self.pred_distance)
        
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
    
    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        for fitting_column, (columns, log_before_scale) in self.scaling_column_groups.items():
                if log_before_scale:
                    df[columns + [fitting_column]] = np.log10(df[columns + [fitting_column]])
                df = self.fit_transform_column(self.scalers[fitting_column], df, fitting_column)
                df = self.transform_columns(self.scalers[fitting_column], df, columns)
        
        self.was_fit = True

        return df.copy()
    
    def fit(self, df:pd.DataFrame):
        for fitting_column, _ in self.scaling_column_groups.items():
                self.fit_on_column(self.scalers[fitting_column], df, fitting_column)
        
        self.was_fit = True

        return
    
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        for fitting_column, (columns, log_before_scale) in self.scaling_column_groups.items():
            if log_before_scale:
                df[columns + [fitting_column]] = np.log10(df[columns + [fitting_column]])
            df = self.transform_columns(self.scalers[fitting_column], df, columns + [fitting_column])
        
        return df.copy()
    
    def inverse_transform_predictions(self, predictions:np.ndarray, fiting_column:str, number_of_columns:int) -> np.ndarray:
        scaler:StandardScaler = self.scalers[fiting_column]
        scaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        if isinstance(scaled_predictions, np.ndarray):
            return scaled_predictions.reshape(-1, number_of_columns)
        else:
            raise ValueError("Predictions should be numpy array")

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

    @staticmethod
    def fit_on_column(scaler:StandardScaler, df:pd.DataFrame, column:str):
        values_to_fit = df[[column]].values
        scaler.fit(values_to_fit)
    
    @staticmethod
    def fit_transform_column(scaler:StandardScaler, df:pd.DataFrame, column:str) -> pd.DataFrame:
        values_to_fit = df[[column]].to_numpy()
        df[[column]]= scaler.fit_transform(values_to_fit)
        return df
    
    @staticmethod
    def transform_columns(scaler:StandardScaler, df:pd.DataFrame, columns:list[str]) -> pd.DataFrame:
        for column in columns:
            values_to_transform = df[[column]].to_numpy()
            transformed_vale = scaler.transform(values_to_transform)
            if isinstance(transformed_vale, np.ndarray):
                df[[column]] = transformed_vale
        
        return df
    
    @staticmethod
    def load_prepared_raw_datasets(ticker_symbvol:str, exchange:str) -> list[pd.DataFrame]:
        logging.info(f"Loading prepared raw datasets for {ticker_symbvol} on {exchange} ...")
        
        dfs:list[pd.DataFrame] = []

        for minute_multiplier in cnts.minute_multipliers.keys():
            dataset_file_name = f"{cnts.data_sets_folder}/{ticker_symbvol}-{exchange}--ib--{minute_multiplier:.0f}--minute--dataset"
            df:pd.DataFrame | None = df_ls.load_df(dataset_file_name)
            if df is not None:
                dfs.append(df)

        return dfs