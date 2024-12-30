import logging

from utils import file_system_utils as fs_utils
from utils import logging_util as log_util
from ib_pipeline import ib_historical_data_downloader as ib_hist_data_dl
from ib_pipeline import ib_raw_data_merger as ib_raw_data_merger


if __name__ == "__main__":
    
    log_util.configure_logging("run_in_pipeline")

    logging.info(f"Starting {__file__} ...")

    fs_utils.create_required_folders()

    logging.info(f"Downloading historical prices ...")
    ib_hist_data_dl.do_step()
    
    logging.info(f"Merging historical data ...")
    ib_raw_data_merger.do_step()
    
    logging.info(f"Finished {__file__} ...")