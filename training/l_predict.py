import lightning as L

import torch
from torch.utils.data import DataLoader
import l_dataset as lds
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt

import l_module as l_module
import l_common as lc
import l_datamodule as ldm

import ib_logging as ib_log
import logging

def transform_predictions_to_numpy(predictions, number_of_variables:int) -> np.ndarray:
    # Concatenate the list of tensors along the first dimension (batch dimension)
    concatenated_tensor = torch.cat(predictions, dim=0)
    
    # Convert the concatenated tensor to a numpy array
    numpy_array = concatenated_tensor.numpy()
    
    # Return the flattened numpy array
    return numpy_array.reshape(-1, number_of_variables)

def predict_and_show_with_expected_values(
    dataloader:DataLoader,
    expected_values:np.ndarray,
    model:l_module.TimeSeriesModule):

    # Create a Trainer instance
    trainer = L.Trainer()
    
        # Use the trainer to predict historical data
    predictions = trainer.predict(model, dataloaders=dataloader)
    
    if predictions is None:
        raise ValueError("No predictions made")

    numpy_predictions = transform_predictions_to_numpy(predictions, len(lc.pred_columns) * 2)
    numpy_predictions_inverse_scaled = data_module.inverse_transform_predictions(
        numpy_predictions, 
        lc.pred_columns[0], 
        len(lc.pred_columns) * 2)
    
    expected_values_inverse_scaled = data_module.inverse_transform_predictions(expected_values, lc.pred_columns[0], len(lc.pred_columns) * 2)
    
    mape = np.mean(np.abs((expected_values_inverse_scaled - numpy_predictions_inverse_scaled) / expected_values_inverse_scaled)) * 100.0
    smape = np.mean(np.abs(expected_values_inverse_scaled - numpy_predictions_inverse_scaled) / (np.abs(expected_values_inverse_scaled) + np.abs(numpy_predictions_inverse_scaled))) * 100.0
    rse = np.sum(np.square(expected_values_inverse_scaled - numpy_predictions_inverse_scaled)) / np.sum(np.square(expected_values_inverse_scaled - np.mean(expected_values_inverse_scaled)))
    
    logging.info(f"Mean Absolute Percentage Error: {mape:.4f}%")
    logging.info(f"Symmetric Mean Absolute Percentage Error: {smape:.4f}%")
    logging.info(f"Relative Squared Error: {rse:.4f}")
    
    plt.plot(numpy_predictions_inverse_scaled, label='Predicted', linestyle='--')
    plt.plot(expected_values_inverse_scaled, label='Expected')
        
    # draw grid
    plt.grid(which='both')
    plt.minorticks_on()
    
    plt.legend()
    plt.show(block = True)

if __name__ == "__main__":

    ib_log.configure_logging("predictiong")

    torch.set_float32_matmul_precision('high')

    model = lc.load_module("lightning_logs/version_368/checkpoints/epoch=14-step=1905.ckpt")
    
    data_module = ldm.StockPriceDataModule (
            "RY", "TSE",
            tail=lc.dataset_tail,
            sequences=lc.sequences,
            pred_columns=lc.pred_columns,
            scaling_column_groups=lc.scaling_column_groups,
            pred_distance=lc.prediction_distance,
            user_tensor_dataset=True,
            batch_size=1024,
            keep_loaded_data=True,
            keep_scaled_data=True,
            keep_validation_dataset=True,
            keep_train_dataset=True
        )

    data_module.prepare_data()

    _, vy = data_module.get_val_src_y()
    predict_and_show_with_expected_values(data_module.val_dataloader(), vy, model)
    
    _, ty = data_module.get_train_src_y()
    predict_and_show_with_expected_values(data_module.train_dataloader_not_shuffled(), ty, model)

