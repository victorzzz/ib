from typing import List
from typing import Dict
from typing import Tuple
from typing import Generator
import constants as cnts

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
    "usa_banks": ("C", "BAC", "JPM", ),
    "tech_software": ("SHOP", "OTEX", "CDAY", "TRI" ),
    "tech_data_processing": ("HUT", "BITF"),
    "transportation": ("TFII", "WCN", "CP", "CNI"),
    "oil_gas": ("TRP", "PBA", "ENB", "SU", "VET", "ERF", "CPG", "OVV", "TECK", "CVE", "CNQ"),
    "usa": ("TSLA", "AAPL", "NVDA", "AMZN", "META", "MSFT", "AMD", "NFLX", "GOOG", "GOOGL"),
    "gold_etf": ("GLD",),
    "etf": ("TDV", "SPY", "DIA"),
    "leveraged_etf" : ("TQQQ", "SQQQ", "QLD"),
    "fixed_income": ("HYHG",)
    """

def get_all_tickers_list() -> List[Tuple[str, Dict[str, int]]]:
    return list(get_all_tickers())

def get_all_tickers() -> Generator[Tuple[str, Dict[str, int]], None, None]:
    for values in ticker_groups.values(): # 
        for key, value in values.items():
            yield (key, value)

def get_test_tickers() -> Generator[Tuple[str, Dict[str, int]], None, None]:
    l = get_all_tickers_list()
    yield l[0]
    yield l[1]

def get_test_tickets_batches(batch_size:int) -> Generator[List[Tuple[str, Dict[str, int]]], None, None]:
    return batch_generator(get_test_tickers(), batch_size)

def get_all_tickets_batches_list(batch_size:int) -> List[List[Tuple[str, Dict[str, int]]]]:
    return list(get_all_tickets_batches(batch_size))

def get_all_tickets_batches(batch_size:int) -> Generator[List[Tuple[str, Dict[str, int]]], None, None]:
    return batch_generator(get_all_tickers(), batch_size)

def batch_generator(sequence:Generator[Tuple[str, Dict[str, int]], None, None], batch_size:int) \
    -> Generator[List[Tuple[str, Dict[str, int]]], None, None]:
    batch = []
    for item in sequence:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_even_items(list:List[Tuple[str, Dict[str, int]]]) -> List[Tuple[str, Dict[str, int]]] :
    return list[::2]

def get_odd_items(list:List[Tuple[str, Dict[str, int]]]) -> List[Tuple[str, Dict[str, int]]] :
    return list[1::2]
