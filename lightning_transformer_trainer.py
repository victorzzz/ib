import sys

import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import model_summary
from watermark import watermark
import lightning.pytorch.loggers as Lloggers
import logging
import ib_logging as ib_log
import lightning_transformer_datamodule as ltdm
import lightning_transformer_model as ltm

if __name__ == "__main__":

    ib_log.configure_logging("lightning_transformer_trainer")

    logging.info(f"Starting {__file__} ...")

    logging.info(watermark(packages="torch,lightning,pandas,numpy,scikit-learn,torchmetrics,pyarrow,ib_insync,pandas_ta,matplotlib"))

    L.seed_everything(4212342)

    # Create data module
    data_module = ltdm.StockTransformerDataModule(
        data[['open', 'close', 'low', 'high', 'ema_open', 'ema_close', 'ema_low', 'ema_high']],
        batch_size=32,
        price_seq_len=64,
        indicator_seq_len=8,
        pred_len=8,
        step=2
    )
    summary = model_summary.ModelSummary(lightning_model, max_depth=-1)
    logging.info(summary)

    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="trading_model_profiler_log")
    trainer = L.Trainer(
        # overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=30,
        accelerator="gpu",
        devices="auto",
        logger=Lloggers.TensorBoardLogger("lightning_logs", name="trading_model"),
        deterministic=True,
        profiler=profiler,
    )
    
    trainer.fit(model=lightning_model, datamodule=dm)
    
    profiler.describe()
    logging.info(profiler.summary())
    
    trainer.test(model=lightning_model, datamodule=dm)