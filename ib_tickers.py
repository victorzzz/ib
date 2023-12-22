from typing import List
from typing import Dict
from typing import Generator
import constants as cnts

ticker_groups = {
    "bank_financial": 
        {
            "RY": (2008980, 4458964),
            "TD": (2009210, 4458978),
            "BNS": (15156975, 4457153), 
            "BMO": (5094, 4458881),
            "CM": (6607659, 4458463),
            "MFC": (1447100, 8165730),
            "SLF": (9662691, 10514248),
            "BAM": (599900805, 600802720)
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
    "leveraged_etf" : ("TQQQ", "QLD"),
    "fixed_income": ("HYHG",)
    """


def get_all_tickers_list() -> List[Dict[str, List[int]]]:
    return list(get_all_tickers())

def get_all_tickers() -> Generator[Dict[str, List[int]], None, None]:
    for values in ticker_groups.values():
        for value in values:
            yield value

def get_test_tickers() -> Generator[Dict[str, List[int]], None, None]:
    l = get_all_tickers_list()
    yield l[0]
    yield l[1]

def get_test_tickets_batches(batch_size:int) -> Generator[List[Dict[str, List[int]]], None, None]:
    return batch_generator(get_test_tickers(), batch_size)

def get_all_tickets_batches_list(batch_size:int) -> List[List[Dict[str, List[int]]]]:
    return list(get_all_tickets_batches(batch_size))

def get_all_tickets_batches(batch_size:int) -> Generator[List[Dict[str, List[int]]], None, None]:
    return batch_generator(get_all_tickers(), batch_size)

def batch_generator(sequence:Generator[Dict[str, List[int]], None, None], batch_size:int) \
    -> Generator[List[Dict[str, List[int]]], None, None]:
    batch = []
    for item in sequence:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch