import sys

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import model_summary
from watermark import watermark
import lightning_datamodule as ldm
import lightning_module as lm
import lightning.pytorch.loggers as Lloggers
import logging
import ib_logging as ib_log

if __name__ == "__main__":

    ib_log.configure_logging("ib_raw_data_merger")

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

    trainer = L.Trainer(
        overfit_batches=1,
        # fast_dev_run=5,
        max_epochs=2,
        accelerator="gpu",
        devices="auto",
        logger=Lloggers.TensorBoardLogger("lightning_logs", name="trading_model"),
        deterministic=True,
    )
    
    trainer.fit(model=lightning_model, datamodule=dm)
    
    trainer.test(model=lightning_model, datamodule=dm)