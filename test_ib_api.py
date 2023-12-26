import time
import datetime as dt

from typing import Tuple
from typing import Dict
from typing import List

from ib_insync import *
import pandas as pd

import ib_tickers as ib_tckrs
import ib_constants as ib_cnts

ib_client:IB = IB()
ib_client.connect(
    readonly=True,
    port=ib_cnts.hist_data_loader_live_port,
    clientId=ib_cnts.hist_data_loader_live_client_id,
    host=ib_cnts.hist_data_loader_live_host)

# contract = Contract(conId=2008980)
contract = Contract(conId=4458964)

ib_client.qualifyContracts(contract)

print(" ")
print(contract)

# date_to = dt.datetime.now() - dt.timedelta(days=4)
date_to = dt.datetime(2020, 12, 8)

final_data_frame:pd.DataFrame = None

for data_type in ib_cnts.hist_data_types:

    print(f"Call {data_type} {contract.conId}--ib--1--minute--{date_to}")

    bars = ib_client.reqHistoricalData(
        contract = contract,
        endDateTime = date_to,
        durationStr = f"10 D",
        barSizeSetting = "1 min",
        whatToShow=data_type,
        useRTH = True
    )

    bars_to_save:List[Dict[str, float]] = None

    if (data_type == "TRADES"):
        bars_to_save = [
            {
                "timestamp": int(dt.datetime.timestamp(bar.date)),
                "TRADES_open": bar.open,
                "TRADES_high": bar.high,
                "TRADES_low": bar.low,
                "TRADES_close": bar.close,
                "TRADES_volume": bar.volume,
                "TRADES_average": bar.average,
                "TRADES_barCount": bar.barCount
            } 
            for bar in bars]
    else:
        bars_to_save = [
            {
                "timestamp": int(dt.datetime.timestamp(bar.date)),
                f"{data_type}_open": bar.open,
                f"{data_type}_high": bar.high,
                f"{data_type}_low": bar.low,
                f"{data_type}_close": bar.close
            } 
            for bar in bars]

    df = pd.DataFrame(bars_to_save).set_index('timestamp')

    if (final_data_frame is None):
        final_data_frame = df
    else:
        final_data_frame = pd.concat([final_data_frame, df], axis=1, sort=True)
        # final_data_frame = final_data_frame.join(df, on='timestamp', how='outer')

    df.to_csv(f"test_data/test_ib_api_{data_type}_{date_to.strftime('%Y-%m-%d')}.csv")

    print(f"{data_type}")
    print(df)

    print("Final")
    final_data_frame.to_csv(f"test_data/FINAL_after_{data_type}_{date_to.strftime('%Y-%m-%d')}.csv")
    print(final_data_frame)
    print("-")