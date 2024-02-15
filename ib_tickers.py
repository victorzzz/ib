from typing import List
from typing import Dict
from typing import Tuple
from typing import Generator
import constants as cnts

"""
ticker_groups = {
    "bank_financial": 
        {
            "RY": { "NYSE": 2008980, "TSX": 4458964 },
            "TD": { "NYSE": 2009210, "TSX": 4458978 },
            "BNS": { "NYSE": 15156975, "TSX": 4457153 }, 
            "BMO": { "NYSE": 5094, "TSX": 4458881 },
            "MFC": { "NYSE": 1447100, "TSX": 8165730 },
            "SLF": { "NYSE": 9662691, "TSX": 10514248 }
        }
    }
"""    

selected_ticker_group_names:List[str] = [
    "bank_financial",
    "usa_banks",
    "tech_software",
    "usa",
    "tech_data_processing",
    "transportation",
    "oil_gas"    
]

ticker_groups:Dict[str, Dict[str, List[str]]] = {
    "bank_financial": 
        {
            "RY": [ "NYSE", "TSE"],
            "TD": [ "NYSE", "TSE"],
            "BNS": [ "NYSE", "TSE"], 
            "BMO": [ "NYSE", "TSE"],
            "MFC": [ "NYSE", "TSE"],
            "SLF": [ "NYSE", "TSE"],
            "CM": [ "NYSE", "TSE"],
            "NA": [ "TSE"],
            "ENB": [ "NYSE", "TSE"]
        },

    "usa_banks": 
        {
            "C" : ["NYSE"], 
            "BAC": ["NYSE"], 
            "JPM": ["NYSE"],
            "GS" : ["NYSE"],
            "MS" : ["NYSE"]
        },

    "tech_software": 
        {
            "SHOP" : ["NYSE", "TSE"],
            "OTEX" : ["NYSE", "TSE"],
            "CDAY" : ["NYSE", "TSE"],
            "TRI" : ["NYSE", "TSE"]
        },

    "tech_data_processing": 
        {
            "HUT": ["NASDAQ", "TSE"], 
            "BITF": ["NASDAQ", "TSE"]
        },

    "transportation": 
        {
            "TFII" : ["NYSE", "TSE"],
            "WCN" : ["NYSE", "TSE"],
            "CP" : ["NYSE", "TSE"],
        },

    "oil_gas": 
        {
            "TRP" : ["NYSE", "TSE"],
            "ENB" : ["NYSE", "TSE"],
            "SU" : ["NYSE", "TSE"],
            "VET" : ["NYSE", "TSE"],
            "ERF" : ["NYSE", "TSE"],
            "OVV" : ["NYSE", "TSE"],
            "CVE" : ["NYSE", "TSE"],
            "CNQ" : ["NYSE", "TSE"],
            "ARX" : ["TSE"]  # ++++++++++
        },

    "usa": 
        {
            "TSLA" : ["NASDAQ"], 
            "AAPL" : ["NASDAQ"], 
            "NVDA" : ["NASDAQ"], 
            "AMZN" : ["NASDAQ"], 
            "META" : ["NASDAQ"], 
            "MSFT" : ["NASDAQ"], 
            "AMD" : ["NASDAQ"], 
            "NFLX" : ["NASDAQ"],
            "INTC" : ["NASDAQ"],
            "CSCO" : ["NASDAQ"],
            "QCOM" : ["NASDAQ"],
            "ADBE" : ["NASDAQ"],
            "PYPL" : ["NASDAQ"],
            "GOOG" : ["NASDAQ"], 
            "GOOGL" : ["NASDAQ"],
            "CRM" : ["NYSE"],
            "V" : ["NYSE"],
            "MA" : ["NYSE"]
        },

    "etf": 
        {
            "SPY" : ["NYSE"], 
            "DIA" : ["NYSE"]
        },
        
        "canadian_etf": 
        {
            "ZSP" : ["TSE"],
            "XIU" : ["TSE"], 
            "XIC" : ["TSE"], 
            "XUU" : ["TSE"], 
            "XRE" : ["TSE"], 
            "XBB" : ["TSE"], 
            "XPF" : ["TSE"], 
            "XIT" : ["TSE"], 
            "XSP" : ["TSE"]
        },

    "leveraged_etf" : 
        {
            "TQQQ" : ["NASDAQ"], 
            "SQQQ" : ["NASDAQ"], 
            "QLD" : ["NYSE"]
        },

        "canadian_leveraged_etf" : 
        {
            "HQU" : ["TSE"],

        },

    "fixed_income": 
        {
            "HYHG" : ["NYSE"]
        },

    "canadian_fixed_income":
        {
            "XHY" : ["TSE"]
        }  
}

forex_tickers = [
    {"USD": "CAD"},
    {"EUR": "CAD"},
    {"GBP": "CAD"}
]

def get_selected_tickers() -> Generator[Tuple[str, List[str]], None, None]:
    g = [(key, ticker_groups[key]) for key in selected_ticker_group_names]
    for group, tickers in g:
       for ticker, exchanges in tickers.items():
           yield (ticker, exchanges)

def get_selected_tickers_list() -> List[Tuple[str, List[str]]]:
    return list(get_selected_tickers())

def get_all_tickers_list() -> List[Tuple[str, List[str]]]:
    return list(get_all_tickers())

def get_all_tickers() -> Generator[Tuple[str, List[str]], None, None]:
    for values in ticker_groups.values():
        for key, value in values.items():
            yield (key, value)

def get_test_tickers() -> Generator[Tuple[str, List[str]], None, None]:
    l = get_all_tickers_list()
    yield l[0]
    yield l[1]

def get_test_tickers_batches(batch_size:int) -> Generator[List[Tuple[str, List[str]]], None, None]:
    return batch_generator(get_test_tickers(), batch_size)

def get_all_tickers_batches_list(batch_size:int) -> List[List[Tuple[str, List[str]]]]:
    return list(get_all_tickers_batches(batch_size))

def get_all_tickers_batches(batch_size:int) -> Generator[List[Tuple[str, List[str]]], None, None]:
    return batch_generator(get_all_tickers(), batch_size)

def get_selected_tickers_batches_list(batch_size:int) -> List[List[Tuple[str, List[str]]]]:
    return list(get_selected_tickers_batches(batch_size))

def get_selected_tickers_batches(batch_size:int) -> Generator[List[Tuple[str, List[str]]], None, None]:
    return batch_generator(get_selected_tickers(), batch_size)

def batch_generator(sequence:Generator[Tuple[str, List[str]], None, None], batch_size:int) \
    -> Generator[List[Tuple[str, List[str]]], None, None]:
    batch = []
    for item in sequence:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_even_items(list:List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]] :
    return list[::2]

def get_odd_items(list:List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]] :
    return list[1::2]
