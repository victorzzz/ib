import pandas as pd
import datetime as dt
import lightning as L
import torch
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint

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
        batch_size=lc.batch_size_param,
        keep_loaded_data=False,
        keep_scaled_data=False,
        keep_validation_dataset=False
    )

    logging.info("Creating model ...")
    model = lc.create_model()

    logging.info("Creating trainer ...")
    
    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="trading_model_profiler_log")
    
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        mode='min',  # 'min' for minimizing the monitored metric
        save_top_k=3,  # Save only the best model
        save_last=True,  # Save the last checkpoint
        enable_version_counter=True
    )
    
    # Train the model
    trainer = L.Trainer(
        # overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=15, 
        log_every_n_steps=10,
        profiler=profiler,
        callbacks=[checkpoint_callback],)
    
    """
    # create learning rate tuner
    tuner = Tuner(trainer)
    
    # Run learning rate finder
    logging.info("Running learning rate finder ...")
    lr_finder = tuner.lr_find(model, datamodule = data_module)
    if lr_finder is None:
        raise ValueError("No lr_finder object returned")
    
    # Results can be found in
    logging.info(f"lr finder result {lr_finder.results}")

    # Plot with
    fig = lr_finder.plot(suggest=True, show=True)
    if fig is None:
        raise ValueError("No figure returned")
    
    # fig.show()
    
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    logging.info(f"Updating learning rate to {new_lr} ...")
    model.hparams.lr = new_lr # type: ignore
    """
    
    logging.info("Fitting model ...")
    trainer.fit(model, data_module)
    
    # Validate the model
    logging.info("Validating model ...")
    trainer.validate(model, data_module)

    """
    # Save the model
    date_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = f"models/final_tr_enc_{date_str}.ckpt"
    logging.info(f"Saving model {model_path}...")

    trainer.save_checkpoint(model_path)

    logging.info(f"Model saved")
    """