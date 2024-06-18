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

# model = TFTModel.load("darts_models/m1") 
model = TFTModel.load_from_checkpoint(
    'tft_8-pred_256-input_128-hidden-size_64-hidden-cont-size_16-heads_3-lstm__2024-06-17-09-31')

print(" ===== model ====")
print(model)

figsize = (9, 6)

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

historical_prediction_val = model.historical_forecasts(
    target_val[:-FORECAST_HORIZON], covar_val[:-FORECAST_HORIZON], future_covar_val, 
    forecast_horizon=FORECAST_HORIZON, stride=FORECAST_HORIZON,
    retrain=False)

historical_prediction_test = model.historical_forecasts(
    target_test[:-FORECAST_HORIZON], covar_test[:-FORECAST_HORIZON], future_covar_test,
    forecast_horizon=FORECAST_HORIZON, stride=FORECAST_HORIZON,
    retrain=False)

"""
historical_prediction_train = model.historical_forecasts(
    target_train[:-FORECAST_HORIZON], covar_train[:-FORECAST_HORIZON], future_covar_train,
    start=0.5, forecast_horizon=FORECAST_HORIZON, stride=FORECAST_HORIZON,
    retrain=False)
"""

mape_on_all_test = mape(target_test, historical_prediction_test)
mape_on_all_val = mape(target_val, historical_prediction_val)
# mape_on_all_train = mape(target_train, historical_prediction_train)

print(f"MAPE on all test set: {mape_on_all_test}")
print(f"MAPE on all val set: {mape_on_all_val}")
# print(f"MAPE on all train set: {mape_on_all_train}")

def plot_all_predictions():
    
    # plot actual series
    plt.figure(figsize=figsize)
    target_test['1m_BID_close'].plot(label="BID CLOSE test")
    target_test['1m_BID_low'].plot(label="BID low test")
    target_test['1m_BID_high'].plot(label="BID high test")
    #target_val['1m_BID_close'].plot(label="BID CLOSE val")
    target_test['1m_ASK_close'].plot(label="ASK CLOSE test")
    target_test['1m_ASK_low'].plot(label="ASK low test")
    target_test['1m_ASK_high'].plot(label="ASK high test")
    #target_val['1m_ASK_close'].plot(label="ASK CLOSE val")
    
    if isinstance(historical_prediction_test, TimeSeries):
        historical_prediction_test['1m_BID_close'].plot(label="pred BID CLOSE test")
        historical_prediction_test['1m_BID_low'].plot(label="pred BID low test")
        historical_prediction_test['1m_BID_high'].plot(label="pred BID high test")
        historical_prediction_test['1m_ASK_close'].plot(label="pred ASK CLOSE test")
        historical_prediction_test['1m_ASK_low'].plot(label="pred ASK low test")
        historical_prediction_test['1m_ASK_high'].plot(label="pred ASK high test")

    #if isinstance(historical_prediction_val, TimeSeries):    
    #    historical_prediction_val['1m_BID_close'].plot(label="pred val")    
    #    historical_prediction_val['1m_ASK_close'].plot(label="pred val")    

    plt.show()

plot_all_predictions()
