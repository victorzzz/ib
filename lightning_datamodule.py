import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torchmetrics
import torch.utils.data as data
import ib_dataset_builder as ib_db

class AmesHousingDataset(data.Dataset):
    def __init__(self, 
        ticker_symbvol:str, 
        exchange:str,
        lock, 
        shared_tickers_cache:dict[str, int]):

        df = ib_db.load_merged_dataframes(ticker_symbvol, exchange, lock, shared_tickers_cache)

        X = df[['Overall Qual',
                'Gr Liv Area',
                'Total Bsmt SF']].values
        y = df['SalePrice'].values

        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

        self.x = torch.tensor(X_std, dtype=torch.float)
        self.y = torch.tensor(y_std, dtype=torch.float).flatten()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
