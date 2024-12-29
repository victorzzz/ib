
import lightning as L
import torch
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, ModelSummary

import matplotlib.pyplot as plt

import ib_logging as ib_log
import logging

import l_model as lm
import l_datamodule as ldm

import l_common as lc

if __name__ == "__main__":

    ib_log.configure_logging("training")

    logging.info(f"Starting {__file__} ...")

    torch.set_float32_matmul_precision('high')

    logging.info("Creating data module ...")
    
    # Create data module
    data_module = ldm.StockPriceDataModule (
        "RY", "TSE",
        tail=lc.dataset_tail,
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
    module = lc.create_module()

    logging.info("Creating trainer ...")
    
    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="trading_model_profiler_log")
    
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        mode='min',  # 'min' for minimizing the monitored metric
        save_top_k=1,  # Save only the best model
        save_last=False,  # Save the last checkpoint
        enable_version_counter=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # gpu_stats = DeviceStatsMonitor()
    
    model_summary = ModelSummary(max_depth=5)
    
    # Train the model
    trainer = L.Trainer(
        # overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=lc.max_epochs_param, 
        log_every_n_steps=10,
        # profiler=profiler,
        callbacks=[checkpoint_callback, lr_monitor, model_summary, ])
    
    logging.info("Fitting model ...")
    trainer.fit(module, data_module)
    
    # Validate the model
    logging.info("Validating model ...")
    trainer.validate(module, data_module)
