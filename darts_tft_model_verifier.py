from typing import Sequence

import datetime as dt

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from darts_tft_model_data_preparation import prepare_traine_val_test_datasets
import matplotlib.pyplot as plt

"""
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
"""

QUANTILES:list[float] = [
    0.05,
    0.1,
    0.5,
    0.9,
    0.95,
]

FORECAST_HORIZON:int = 8  # 1 day of 1m data
INPUT_CHUNK_LENGTH:int = FORECAST_HORIZON * 30 

 # reproducibility
torch.manual_seed(42)

(target_train, target_val, target_test), (covar_train, covar_val, covar_test), (future_covar_train, future_covar_val, future_covar_test), price_scaler, val_scaler = prepare_traine_val_test_datasets("RY", "TSE", tail=0.2)

model = TFTModel.load("tft_model_final_8-pred_240-input_256-hiddensize_64-heads_8-lstm__2024-06-16-17-36") 

print(" ===== model ====")
print(model)

if isinstance(model, TFTModel):

    series = target_train[:-FORECAST_HORIZON]
    covariates = covar_train[:-FORECAST_HORIZON]
    future_covariates = future_covar_train

    explainer = TFTExplainer(
        model,
        background_series=series,
        background_past_covariates=covariates,
        background_future_covariates=future_covariates)
    
    explainability_result = explainer.explain()
    print(" ===== explainability_result ====")
    print(explainability_result)

test_length = len(target_test)

all_test_predictions:TimeSeries | Sequence[TimeSeries] | None = None
all_val_predictions:TimeSeries | Sequence[TimeSeries] | None = None


for i in range(0, test_length - INPUT_CHUNK_LENGTH - FORECAST_HORIZON - 1, FORECAST_HORIZON):
    target_test_chunk = target_test[i:i+INPUT_CHUNK_LENGTH + 1]
    covar_test_chunk = covar_test[i:i+INPUT_CHUNK_LENGTH + 1]
    future_covar_test_chunk = future_covar_test[i:i+INPUT_CHUNK_LENGTH+FORECAST_HORIZON+1]
    
    target_val_chunk = target_val[i:i+INPUT_CHUNK_LENGTH + 1]
    covar_val_chunk = covar_val[i:i+INPUT_CHUNK_LENGTH + 1]
    future_covar_val_chunk = future_covar_val[i:i+INPUT_CHUNK_LENGTH+FORECAST_HORIZON+1]
    
    prediction_on_test_chunk = model.predict(FORECAST_HORIZON, target_test_chunk, covar_test_chunk, future_covar_test_chunk) 
    prediction_on_val_chunk = model.predict(FORECAST_HORIZON, target_val_chunk, covar_val_chunk, future_covar_val_chunk)
    
    if all_test_predictions is None:
        all_test_predictions = prediction_on_test_chunk
    elif isinstance(all_test_predictions, TimeSeries) and isinstance(prediction_on_test_chunk, TimeSeries):
        all_test_predictions = all_test_predictions.append(prediction_on_test_chunk)    
    
    if all_val_predictions is None:
        all_val_predictions = prediction_on_val_chunk
    elif isinstance(all_val_predictions, TimeSeries) and isinstance(prediction_on_val_chunk, TimeSeries):
        all_val_predictions = all_val_predictions.append(prediction_on_val_chunk)

prediction_on_last_test = model.predict(FORECAST_HORIZON, target_test[:-FORECAST_HORIZON], covar_test[:-FORECAST_HORIZON], future_covar_test) #, predict_likelihood_parameters=True)
prediction_on_last_val = model.predict(FORECAST_HORIZON, target_val[:-FORECAST_HORIZON], covar_val[:-FORECAST_HORIZON], future_covar_val) #, predict_likelihood_parameters=True)

if isinstance(prediction_on_last_test, TimeSeries) and isinstance(prediction_on_last_val, TimeSeries):

    mape_on_last_test = mape(target_test, prediction_on_last_test)
    mape_on_last_val = mape(target_val, prediction_on_last_val)

    print(f"MAPE on last test set: {mape_on_last_test}")
    print(f"MAPE on last val set: {mape_on_last_val}")

mape_on_all_test = mape(target_test, all_test_predictions)
mape_on_all_val = mape(target_val, all_val_predictions)

print(f"MAPE on all test set: {mape_on_all_test}")
print(f"MAPE on all val set: {mape_on_all_val}")

figsize = (9, 6)

def plot_all_predictions():
    
    # plot actual series
    plt.figure(figsize=figsize)
    target_test.plot(label="actual test")
    target_val.plot(label="actual val")
    
    if isinstance(all_test_predictions, TimeSeries):
        all_test_predictions.plot(label="prediction on test")

    if isinstance(all_val_predictions, TimeSeries):    
        all_val_predictions.plot(label="prediction on val")    

    plt.show()

plot_all_predictions()
