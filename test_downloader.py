import ib_historical_data_downloader as ib_hdd
import ib_constants as ib_cnts
import ib_tickers as ib_tckrs

tickers = ib_tckrs.get_all_tickers_list()
ib_hdd.download_stock_bars_for_tickers(
    tickers[-1:],
    ib_cnts.hist_data_loader_live_port,
    ib_cnts.hist_data_loader_live_client_id,
    ib_cnts.hist_data_loader_live_host
)