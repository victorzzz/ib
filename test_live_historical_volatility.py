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
date_to = dt.datetime(2024, 1, 30)

ib_client.reqHistoricalData(
        contract = contract,
        endDateTime = date_to,
        durationStr = f"10 D",
        barSizeSetting = "1 min",
        whatToShow='HISTORICAL_VOLATILITY',
        useRTH = True,
        keepUpToDate=True
    )