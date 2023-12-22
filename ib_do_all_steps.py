import ib_historical_data_downloader as hist_downloader
import ib_seconds_historical_data_downloader as seconds_hist_downloader

if __name__ == "__main__": 
    
    print("IB routing has been started")

    # download historical data
    hist_downloader.do_step() 

    # download seconds historical data
    seconds_hist_downloader.do_step() 

    print("IB routing has been done")
