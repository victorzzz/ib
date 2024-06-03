import sys

import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import model_summary
from watermark import watermark
import lightning_datamodule as ldm
import lightning_module as lm
import lightning.pytorch.loggers as Lloggers
import logging
import ib_logging as ib_log

if __name__ == "__main__":

    ib_log.configure_logging("lightning_trading_model_trainer")

    logging.info(f"Starting {__file__} ...")

    logging.info(watermark(packages="torch,lightning,pandas,numpy,scikit-learn,torchmetrics,pyarrow,ib_insync,pandas_ta,matplotlib"))

    L.seed_everything(123)
    dm = ldm.HistoricalMarketDataDataModule("RY", "TSE")

    pytorch_model = lm.PyTorchTradingModel()
    lightning_model = lm.LightningTradingModel(
        model = pytorch_model, 
        learning_rate = 0.15
    )

    summary = model_summary.ModelSummary(lightning_model, max_depth=-1)
    logging.info(summary)

    profiler = AdvancedProfiler(dirpath="profiler_logs", filename="trading_model_profiler_log")
    trainer = L.Trainer(
        # overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=10,
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