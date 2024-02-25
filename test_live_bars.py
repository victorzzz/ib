import time
import datetime as dt

from typing import Optional

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
    print (f"current time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"onBarUpdate BarDataList: {bars.contract.symbol} {bars.whatToShow} ")
    if (hasNewBar):
        print("onBarUpdate Bar 0: ", bars[0])
        print(f"onBarUpdate BarDataList: {bars.contract.symbol} {bars.whatToShow} ")
        print(f"onBarUpdate Bar -1: {bars[-1].date:%Y-%m-%d %H:%M:%S}", bars[-1])
        print(f"onBarUpdate Bar -2: {bars[-2].date:%Y-%m-%d %H:%M:%S}", bars[-2])

def onRealTimeBarUpdate (realTimeBars: RealTimeBarList, hasNewBar: bool):
    print("onRealTimeBarUpdate realTimeBars: ", len(realTimeBars))
    print("onRealTimeBarUpdate Has new bar: ", hasNewBar)
    print(f"current time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"onRealTimeBarUpdate RealTimeBarList: {realTimeBars.contract.symbol} {realTimeBars.whatToShow} ")
    if (hasNewBar):
        print("onRealTimeBarUpdate Bar 0: ", realTimeBars[0])
        print(f"onRealTimeBarUpdate RealTimeBarList: {realTimeBars.contract.symbol} {realTimeBars.whatToShow} ")
        print(f"onRealTimeBarUpdate Bar -1: {realTimeBars[-1].time:%Y-%m-%d %H:%M:%S}", realTimeBars[-1])

def onUpdate ():
    print("onUpdate Bars:", 0 if bars is None else len(bars))

ib_log.configure_logging(ib_sync_log_level=logging.DEBUG)

ib_client:IB = IB()
ib_client.connect(
    readonly=True,
    port=ib_cnts.hist_data_loader_live_port,
    clientId=ib_cnts.hist_data_loader_live_client_id,
    host=ib_cnts.hist_data_loader_live_host)

contract = Contract(conId=2008980)
contract2 = Contract(conId=2009210)
contract3 = Contract(conId=15156975)
contract4 = Contract(conId=5094)
contract5 = Contract(conId=1447100)

# contract = Contract(conId=4458964)

ib_client.qualifyContracts(contract)
ib_client.qualifyContracts(contract2)
ib_client.qualifyContracts(contract3)
ib_client.qualifyContracts(contract4)
ib_client.qualifyContracts(contract5)


print(" ")
print(contract)

# date_to = dt.datetime.now() - dt.timedelta(days=4)
date_to = dt.datetime(2024, 2, 3)

print(1)
bars = ib_client.reqHistoricalData(
        contract = contract,
        endDateTime = "",
        durationStr = "240 S",
        barSizeSetting = "1 min",
        whatToShow='ASK',
        useRTH = True,
        keepUpToDate=True
    )

print("bars size after initial download: ", len(bars))

print(2)
bars2 = ib_client.reqHistoricalData(
        contract = contract,
        endDateTime = "",
        durationStr = "60 S",
        barSizeSetting = "15 secs",
        whatToShow='BID',
        useRTH = True,
        keepUpToDate=True
    )

print("bars 2 size after initial download: ", len(bars2))

print(3)
bars3 = ib_client.reqHistoricalData(
        contract = contract3,
        endDateTime = "",
        durationStr = "240 S",
        barSizeSetting = "30 secs",
        whatToShow='TRADES',
        useRTH = True,
        keepUpToDate=True
    )

print("bars 3 size after initial download: ", len(bars3))

print(4)
bars4 = ib_client.reqHistoricalData(
        contract = contract4,
        endDateTime = "",
        durationStr = "240 S",
        barSizeSetting = "30 secs",
        whatToShow='BID',
        useRTH = True,
        keepUpToDate=True
    )

print("bars 4 size after initial download: ", len(bars4))

print(5)
bars5 = ib_client.reqHistoricalData(
        contract = contract5,
        endDateTime = "",
        durationStr = "240 S",
        barSizeSetting = "30 secs",
        whatToShow='MIDPOINT',
        useRTH = False,
        keepUpToDate=True
    )

print("bars 5 size after initial download: ", len(bars5))

ib_client.pendingTickersEvent += onPendingTickers
ib_client.barUpdateEvent += onBarUpdate

newsProviders = ib_client.reqNewsProviders()
print("NewsProviders: ", newsProviders)

# realTimeBars = ib_client.reqRealTimeBars(contract, 5, "BID", False)
# realTimeBars.updateEvent += onRealTimeBarUpdate

# ib_client.updateEvent += onUpdate

ib_client.run()