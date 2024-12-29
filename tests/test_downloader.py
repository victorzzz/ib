import ib_historical_data_downloader as ib_hdd
import ib_constants as ib_cnts
import ib_tickers as ib_tckrs

import multiprocessing
from typing import Sequence

if __name__ == "__main__":

    # tickers = ib_tckrs.get_all_tickers_list()

    #tickers_to_test = tickers[1:2]
    
    tickers_to_test: list[tuple[str, list[str]]] = [("RY", ["TSE"],),]

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()

    shared_dict = (dict[str, int])(manager.dict())

    ib_hdd.download_stock_bars_for_tickers(
        tickers_to_test,
        ib_cnts.hist_data_loader_live_port,
        ib_cnts.hist_data_loader_live_client_id,
        ib_cnts.hist_data_loader_live_host,
        lock,
        shared_dict
    )