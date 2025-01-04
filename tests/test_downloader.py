from ib_pipeline import ib_historical_data_downloader as ib_hdd
from ib_pipeline import ib_constants as ib_cnts

import multiprocessing

def test_ib_historical_data_downloader():
        
    tickers_to_test: list[tuple[str, list[str]]] = [("GRID", ["TSE"],),]

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