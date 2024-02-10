import time
import datetime as dt

from typing import Tuple
from typing import Optional
from typing import Dict
from typing import List

from ib_insync import *
import pandas as pd
import date_time_utils as dt_utils

import ib_tickers as ib_tckrs
import ib_constants as ib_cnts
import ib_logging as ib_log
import logging

bars:Optional[BarDataList] = None

def onPendingTickers(tickers):
    print("Pending tickers: ", tickers)

def onBarUpdate (bars: BarDataList, hasNewBar: bool):
    print("onBarUpdate Bars: ", len(bars))
    print("onBarUpdate Has new bar: ", hasNewBar)

def onUpdate ():
    print("onUpdate Bars:", 0 if bars is None else len(bars))

ib_log.configure_logging(ib_sync_log_level=logging.DEBUG)

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
date_to = dt.datetime(2024, 2, 3)

bars = ib_client.reqHistoricalData(
        contract = contract,
        endDateTime = "",
        durationStr = "3600 S",
        barSizeSetting = "10 secs",
        whatToShow='TRADES',
        useRTH = True,
        keepUpToDate=True
    )

print("bars size after initial download: ", len(bars))

ib_client.pendingTickersEvent += onPendingTickers
ib_client.barUpdateEvent += onBarUpdate
ib_client.updateEvent += onUpdate

ib_client.run()