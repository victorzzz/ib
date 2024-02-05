import logging
import ib_logging as ib_log

def do_step():
    ib_log.configure_logging("ib_seconds_historical_data_downloader")

    logging.info(f"Starting {__file__} ...") 

if __name__ == "__main__":
    do_step()