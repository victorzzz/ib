import datetime as dt

from typing import Optional

from ib_insync import IB, Contract
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

usd_contract = Contract(symbol="SHOP", secType="STK", exchange="NYSE")
usd_contract_details = ib_client.reqContractDetails(usd_contract)

usd_df = pd.DataFrame([vars(c) for c in usd_contract_details])
usd_df.to_csv("shop_usd_contract_details_2.csv")
ib_client.sleep(3)

cad_contract = Contract(symbol="SHOP", secType="STK", exchange="TSE")
cad_contract_details = ib_client.reqContractDetails(cad_contract)
cad_df = pd.DataFrame([vars(c) for c in cad_contract_details])
cad_df.to_csv("shop_cad_contract_details_2.csv")
ib_client.sleep(3)

usdcad_contract = Contract(symbol="USD", secType="CASH", currency="CAD", exchange="IDEALPRO")
usdcad_contract_details = ib_client.reqContractDetails(usdcad_contract)
usdcad_df = pd.DataFrame([vars(c) for c in usdcad_contract_details])
usdcad_df.to_csv("usdcad_contract_details_2.csv")
ib_client.sleep(3)

ib_client.disconnect()