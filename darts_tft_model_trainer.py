import datetime as dt

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape, mase, smape, mse, rmse, ope, mae
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from darts_tft_model_data_preparation import prepare_traine_val_test_datasets

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
"""
QUANTILES:list[float] = [
    0.05,
    0.1,
    0.5,
    0.9,
    0.95,
]
"""

FORECAST_HORIZON:int = 8  
INPUT_CHUNK_LENGTH:int = FORECAST_HORIZON * 32 

MODEL_HIDDEN_SIZE:int = 128
MODEL_HIDDEN_CONTINUOUS_SIZE:int = 64
MODEL_LSTM_LAYERS:int = 3
MODEL_ATTENTION_HEADS:int = 16
MODEL_BATCH_SIZE:int = 192
MODEL_DROP_OUT:float = 0.06

 # reproducibility
torch.manual_seed(42)

torch.set_float32_matmul_precision('medium')

# throughout training we'll monitor the validation loss for early stopping
early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [early_stopper, lr_monitor]

pl_trainer_kwargs = {"callbacks": callbacks}

date = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
model_name = f"tft_{FORECAST_HORIZON}-pred_{INPUT_CHUNK_LENGTH}-input_{MODEL_HIDDEN_SIZE}-hidden-size_{MODEL_HIDDEN_CONTINUOUS_SIZE}-hidden-cont-size_{MODEL_ATTENTION_HEADS}-heads_{MODEL_LSTM_LAYERS}-lstm__{date}"

model:TFTModel = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,
    hidden_size=MODEL_HIDDEN_SIZE,
    lstm_layers=MODEL_LSTM_LAYERS,
    num_attention_heads=MODEL_ATTENTION_HEADS,
    dropout=MODEL_DROP_OUT,
    hidden_continuous_size=MODEL_HIDDEN_CONTINUOUS_SIZE,
    batch_size=MODEL_BATCH_SIZE,
    n_epochs=10,
    add_relative_index=True,
    full_attention = True,
    add_encoders=None,
    use_static_covariates=False,
    #likelihood=QuantileRegression(
    #    quantiles=QUANTILES
    #),  # QuantileRegression is set per default
    loss_fn=torch.nn.MSELoss(),
    # pl_trainer_kwargs=pl_trainer_kwargs,
    log_tensorboard=True,
    save_checkpoints=True,
    random_state=42,
    model_name=model_name,
)

(target_train, target_val, target_test), (covar_train, covar_val, covar_test), (future_covar_train, future_covar_val, future_covar_test), price_scaler, val_scaler = prepare_traine_val_test_datasets("RY", "TSE", tail=0.2)

model.fit(
    target_train[:-FORECAST_HORIZON], 
    past_covariates = covar_train[:-FORECAST_HORIZON], 
    future_covariates = future_covar_train,
    val_series = target_val[:-FORECAST_HORIZON],
    val_past_covariates = covar_val[:-FORECAST_HORIZON],
    val_future_covariates = future_covar_val,
    verbose=True)

model.save(f"darts_models/final_{model_name}")

if isinstance(model, TFTModel):

    explainer = TFTExplainer(
        model,
        background_series=target_train[:-FORECAST_HORIZON],
        background_past_covariates=covar_train[:-FORECAST_HORIZON],
        background_future_covariates=future_covar_train)
    
    explainability_result = explainer.explain()
    print(" ===== explainability_result ====")
    print(explainability_result)

prediction_on_test = model.predict(FORECAST_HORIZON, target_test[:-FORECAST_HORIZON], covar_test[:-FORECAST_HORIZON], future_covar_test)
prediction_on_val = model.predict(FORECAST_HORIZON, target_val[:-FORECAST_HORIZON], covar_val[:-FORECAST_HORIZON], future_covar_val)

mape_on_test = mape(target_test, prediction_on_test)
mape_on_val = mape(target_val, prediction_on_val)

smape_on_test = smape(target_test, prediction_on_test)
smape_on_val = smape(target_val, prediction_on_val)

ope_on_test = ope(target_test, prediction_on_test)
ope_on_val = ope(target_val, prediction_on_val)

rmse_on_test = rmse(target_test, prediction_on_test)
rmse_on_val = rmse(target_val, prediction_on_val)

print(f"MAPE on test set: {mape_on_test}")
print(f"MAPE on val set: {mape_on_val}")
print(f"SMAPE on test set: {smape_on_test}")
print(f"SMAPE on val set: {smape_on_val}")
print(f"OPE on test set: {ope_on_test}")
print(f"OPE on val set: {ope_on_val}")
print(f"RMSE on test set: {rmse_on_test}")
print(f"RMSE on val set: {rmse_on_val}")