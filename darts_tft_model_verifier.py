from typing import Sequence

import datetime as dt

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape, smape, ope
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from darts_tft_model_data_preparation import prepare_train_val_test_datasets, inverse_transform

figsize = (9, 6)

def predict(model:TFTModel, target:TimeSeries, covar:TimeSeries, future_covar:TimeSeries, horizon:int, input_length:int, n:int) -> list[TimeSeries]:
    
    if not isinstance(model, TFTModel) or not isinstance(target, TimeSeries) or not isinstance(covar, TimeSeries) or not isinstance(future_covar, TimeSeries):
        raise ValueError("Invalid input types")
    
    result_step = horizon // n
    
    result:list[TimeSeries] = []
    
    for glogal_shift in range(0, horizon, result_step):
    
        target_chanks = []
        covar_chanks = []
        future_covar_chanks = []
    
        for i in range(0, len(target) - input_length - horizon * 2, horizon):
            target_chunk = target[i + glogal_shift:i + glogal_shift + input_length]
            covar_chunk = covar[i + glogal_shift:i  + glogal_shift + input_length]
            future_covar_chunk = future_covar[i + glogal_shift:i + glogal_shift + input_length + horizon]

            target_chanks.append(target_chunk)
            covar_chanks.append(covar_chunk)
            future_covar_chanks.append(future_covar_chunk)

        prediction_timesries_list:TimeSeries | Sequence[TimeSeries] = model.predict(horizon, target_chanks, covar_chanks, future_covar_chanks)
        
        if isinstance(prediction_timesries_list, TimeSeries):
            raise ValueError("Invalid predictions type")
        else:
             predictions = concatenate(prediction_timesries_list, axis='time')
            
        result.append(predictions)
    
    return result

if __name__ == "__main__":

    # reproducibility
    torch.manual_seed(42)

    (target_train, target_val, target_test), (covar_train, covar_val, covar_test), (future_covar_train, future_covar_val, future_covar_test), price_scaler, vol_scaler = prepare_train_val_test_datasets("RY", "TSE", tail=0.3)

    model = TFTModel.load("FromLambda/darts_models/final_tft_16-pred_512-input_256-hidden-size_64-hidden-cont-size_32-heads_3-lstm__2024-06-22-14-22") 
    #model = TFTModel.load_from_checkpoint('tft_8-pred_256-input_128-hidden-size_64-hidden-cont-size_16-heads_3-lstm__2024-06-17-09-31')

    print(" ===== model ====")
    print(model)

    if not isinstance(model, TFTModel):
        exit("Model is not a TFTModel")

    horizon = model.output_chunk_length

    series = target_train[:-horizon]
    covariates = covar_train[:-horizon]
    future_covariates = future_covar_train

    explainer = TFTExplainer(
        model,
        background_series=series,
        background_past_covariates=covariates,
        background_future_covariates=future_covariates)
    
    explainability_result = explainer.explain()

    """
    historical_prediction_val = model.historical_forecasts(
        target_val[:-horizon], covar_val[:-horizon], future_covar_val, 
        forecast_horizon=4, stride=4,   #horizon // 4,
        # last_points_only=True, 
        retrain=False)
    """
    
    pred_length = len(target_val)
    
    target_val = target_val.tail(pred_length)
    covar_val = covar_val.tail(pred_length)
    future_covar_val = future_covar_val.tail(pred_length)
    
    historical_prediction_val:list[TimeSeries] = predict(model, target_val, covar_val, future_covar_val, horizon, model.input_chunk_length, 2)
    
    target_for_metrics = target_val[model.input_chunk_length:]
    predictions_for_metrics = historical_prediction_val[:len(target_for_metrics)]
    
    mape_on_all_val_0 = mape(target_for_metrics, predictions_for_metrics[0])
    smape_on_all_val_0 = smape(target_for_metrics, predictions_for_metrics[0])
    ope_on_all_val_0 = ope(target_for_metrics, predictions_for_metrics[0])

    mape_on_all_val_1 = mape(target_for_metrics, predictions_for_metrics[1])
    smape_on_all_val_1 = smape(target_for_metrics, predictions_for_metrics[1])
    ope_on_all_val_1 = ope(target_for_metrics, predictions_for_metrics[1])

    print(f"MAPE on all val set: {mape_on_all_val_0}")
    print(f"SMAPE on all val set: {smape_on_all_val_0}")
    print(f"OPE on all val set: {ope_on_all_val_0}")

    print(f"MAPE on all val se 1t: {mape_on_all_val_1}")
    print(f"SMAPE on all val set 1: {smape_on_all_val_1}")
    print(f"OPE on all val set 1: {ope_on_all_val_1}")

  
    historical_prediction_val_0 = inverse_transform(historical_prediction_val[0], price_scaler, vol_scaler)
    historical_prediction_val_1 = inverse_transform(historical_prediction_val[1], price_scaler, vol_scaler)
    target_val = inverse_transform(target_val, price_scaler, vol_scaler)
    
    # plot actual series
    plt.figure(figsize=figsize)
    target_val['1m_BID_close'].plot(label="BID CLOSE val")
    target_val['1m_ASK_close'].plot(label="ASK CLOSE val")

    historical_prediction_val_0['1m_BID_close'].plot(label="BID CLOSE pred 0")
    historical_prediction_val_0['1m_ASK_close'].plot(label="ASK CLOSE pred 0")

    historical_prediction_val_1['1m_BID_close'].plot(label="BID CLOSE pred 1")
    historical_prediction_val_1['1m_ASK_close'].plot(label="ASK CLOSE pred 1")

    plt.show()


