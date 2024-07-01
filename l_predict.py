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

    torch.set_float32_matmul_precision('medium')

    model = lc.load_model("models/final_tr_enc_2024-07-01-15-06.ckpt")

    data_module = ldm.StockPriceDataModule (
            "RY", "TSE",
            sequences=lc.sequences,
            pred_columns=lc.pred_columns,
            scaling_column_groups=lc.scaling_column_groups,
            pred_distance=lc.prediction_distance,
            user_tensor_dataset=True,
            batch_size=128,
            keep_loaded_data=True,
            keep_scaled_data=True
        )

    data_module.prepare_data()

    """
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
    """
    
    df = data_module.get_df()
    
    train_border = int(0.8*len(df))

    history_len = max(seq[0] for seq in lc.sequences)
        
    # train_data:pd.DataFrame = data[:train_border]
    val_data:pd.DataFrame = df[train_border + history_len + lc.prediction_distance:]
    actual_values = val_data[lc.pred_columns].to_numpy()
    
    _, scaled_val_data = data_module.get_scaled_dfs()
    
    plt.plot(scaled_val_data['1m_MIDPOINT_close'], label='Scaled 1m_MIDPOINT_close')
    plt.plot(scaled_val_data['1m_TRADES_average'], label='Scaled 1m_TRADES_average')
    plt.plot(scaled_val_data['1m_TRADES_volume'], label='Scaled 1m_TRADES_volume')
    plt.legend()
    plt.show(block = True)
    
    """
    plt.plot(numpy_predictions[:, 0], label='NOT Inverse Scalled (from model) Predicted 0')
    plt.plot(numpy_predictions_inverse_scaled[:, 0], label='Predicted 0')
    #plt.plot(numpy_predictions_inverse_scaled[:, 1], label='Predicted 1')
    
    plt.plot(actual_values[:, 0], label='Actual 0')
    #plt.plot(actual_values[:, 1], label='Actual 1')
    
    plt.plot(scaled_val_data[lc.pred_columns[0]], label='Scaled actual data')
    
    plt.legend()
    plt.show(block = True)
    """
