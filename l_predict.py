import lightning as L

import torch
from torch.utils.data import DataLoader
import l_dataset as lds
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt

import l_model as lm
import l_common as lc
import l_datamodule as ldm

import ib_logging as ib_log
import logging

def transform_predictions_to_numpy(predictions, number_of_variables:int) -> np.ndarray:
    """
    Transforms the list of tensors (predictions) into a single numpy array.
    
    Parameters:
    predictions (list of torch.Tensor): List of prediction tensors where each tensor has shape [batch_size, 1].
    
    Returns:
    np.ndarray: Numpy array containing all predictions in the same order.
    """
    # Concatenate the list of tensors along the first dimension (batch dimension)
    concatenated_tensor = torch.cat(predictions, dim=0)
    
    # Convert the concatenated tensor to a numpy array
    numpy_array = concatenated_tensor.numpy()
    
    # Return the flattened numpy array
    return numpy_array.reshape(-1, number_of_variables)

if __name__ == "__main__":

    ib_log.configure_logging("predictiong")

    torch.set_float32_matmul_precision('medium')

    model = lc.load_model("lightning_logs/version_82/checkpoints/epoch=9-step=15130.ckpt")

    data_module = ldm.StockPriceDataModule (
            "RY", "TSE",
            sequences=lc.sequences,
            pred_columns=lc.pred_columns,
            scaling_column_groups=lc.scaling_column_groups,
            pred_distance=lc.prediction_distance,
            user_tensor_dataset=True,
            batch_size=128,
            keep_loaded_data=True,
            keep_scaled_data=True,
            keep_validation_dataset=True
        )

    # Create a Trainer instance
    trainer = L.Trainer()

    # Use the trainer to predict historical data
    historical_predictions = trainer.predict(model, datamodule=data_module)

    if historical_predictions is None:
        raise ValueError("No predictions made")

    numpy_predictions = transform_predictions_to_numpy(historical_predictions, len(lc.pred_columns))
    numpy_predictions_inverse_scaled = data_module.inverse_transform_predictions(
        numpy_predictions, 
        lc.pred_columns[0], 
        len(lc.pred_columns))

    df, train_df, val_df = data_module.get_df()
    
    scaled_train_df, scaled_val_df = data_module.get_scaled_dfs()
    
    val_x, val_y = data_module.get_val_src_y()
    
    val_y_inverse_scaled = data_module.inverse_transform_predictions(val_y, lc.pred_columns[0], len(lc.pred_columns))
    
    history_len = max(seq[0] for seq in lc.sequences)
    actual_values = val_df[history_len + lc.prediction_distance:][lc.pred_columns].to_numpy()
    
    mape = np.mean(np.abs((actual_values - numpy_predictions_inverse_scaled) / actual_values)) * 100.0
    smape = np.mean(np.abs(actual_values - numpy_predictions_inverse_scaled) / (np.abs(actual_values) + np.abs(numpy_predictions_inverse_scaled))) * 100.0
    rse = np.sum(np.square(actual_values - numpy_predictions_inverse_scaled)) / np.sum(np.square(actual_values - np.mean(actual_values)))
    
    logging.info(f"Mean Absolute Percentage Error: {mape:.4f}%")
    logging.info(f"Symmetric Mean Absolute Percentage Error: {smape:.4f}%")
    logging.info(f"Relative Squared Error: {rse:.4f}")
    
    # plt.plot(numpy_predictions[:, 0], label='Predicted from model')
    plt.plot(numpy_predictions_inverse_scaled[:, 0], label='Predicted inverse scaled')
        
    # plt.plot(actual_scalled_midpoint_close, label='actual_scalled_midpoint_close')
    plt.plot(val_y_inverse_scaled, label='val_y_inverse_scaled', linewidth=5)
    
    plt.plot(actual_values, label='Actual 1m_MIDPOINT_close')
    # plt.plot(actual_volume, label='Actual 1m_TRADES_volume')
    
    # draw grid
    plt.grid(which='both')
    plt.minorticks_on()
    
    plt.legend()
    plt.show(block = True)

