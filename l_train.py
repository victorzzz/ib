import pandas as pd
import datetime as dt
import lightning as L
import torch
import matplotlib.pyplot as plt

import ib_logging as ib_log
import logging

import l_model as lm
import l_datamodule as ldm

import l_common as lc

if __name__ == "__main__":

    ib_log.configure_logging("training")

    logging.info(f"Starting {__file__} ...")

    torch.set_float32_matmul_precision('medium')

    logging.info("Creating data module ...")
    # Create data module
    data_module = ldm.StockPriceDataModule (
        "RY", "TSE",
        sequences=lc.sequences,
        pred_columns=lc.pred_columns,
        scaling_column_groups=lc.scaling_column_groups,
        pred_distance=lc.prediction_distance,
        user_tensor_dataset=True,
        batch_size=128,
        keep_loaded_data=True
    )

    logging.info("Creating model ...")
    model = lc.create_model()

    logging.info("Creating trainer ...")
    # Train the model
    trainer = L.Trainer(
        # overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=2, 
        log_every_n_steps=10)
    
    logging.info("Fitting model ...")
    trainer.fit(model, data_module)

    logging.info("Training completed.")
    df = data_module.get_df()

    plt.plot(df['1m_MIDPOINT_close'], label='1m_MIDPOINT_close')
    plt.show(block=True)

    # Validate the model
    logging.info("Validating model ...")
    trainer.validate(model, data_module)

    # Save the model
    date_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = f"models/final_tr_enc_{date_str}.ckpt"
    logging.info(f"Saving model {model_path}...")

    trainer.save_checkpoint(model_path)

    logging.info(f"Model saved")