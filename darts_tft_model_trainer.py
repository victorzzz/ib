import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from darts_tft_model_data_preparation import prepare_traine_val_test_datasets

QUANTILES:list[float] = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]

FORECAST_HORIZON:int = 390  # 1 day of 1m data
INPUT_CHUNK_LENGTH:int = FORECAST_HORIZON * 5 

model:TFTModel = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=30,
    add_relative_index=True,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=QUANTILES
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    log_tensorboard=True,
    random_state=42,
)

"""
# Callbacks
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='step')

# Trainer
trainer = Trainer(
    callbacks=[early_stop_callback, lr_monitor],
    max_epochs=100,
    log_every_n_steps=50,
)
"""

(target_train, target_val, target_test), (covar_train, covar_val, covar_test), price_scaler, val_scaler = prepare_traine_val_test_datasets("RY", "TSE", tail=0.1)

model.fit(
    target_train, past_covariates = covar_train, 
    val_series = target_val, val_past_covariates = covar_val,
    verbose=True)