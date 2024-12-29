import logging
from ..utils import logging_util as log_util

def do_step():
    log_util.configure_logging("ib_seconds_historical_data_downloader")

    logging.info(f"Starting {__file__} ...") 

if __name__ == "__main__":
    do_step()