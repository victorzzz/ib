
from typing import Generator
from common import constants as cnts

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

selected_ticker_group_names:list[str] = [
    "bank_financial",
    #"minerals",
    "tech_software",
    #"oil_gas",
    #"energy",
    #"tech_data_processing",
    #"transportation",        
    #"canadian_etf",
    #"canadian_leveraged_etf",
    # "usa",
]

ticker_groups:dict[str, dict[str, list[str]]] = {
    "bank_financial": 
        {
            "RY": [ "TSE"], # "NYSE"
            "TD": [ "TSE"], # "NYSE"
            "BNS": [ "TSE"], # +++++ # "NYSE" # !
            "BMO": [  "TSE"], # +++++ # "NYSE"
            "MFC": [  "TSE"], # +++++ # "NYSE"
            "SLF": [  "TSE"], # "NYSE"
            "CM": [  "TSE"], # +++++ # "NYSE"
            "NA": [ "TSE"],
            "ENB": [ "TSE"] , # "NYSE"
            
            "FC": [ "TSE"], # !
            "AI": [ "TSE"], # !
            
            "GLXY": [ "TSE"], # !?
            
            "AD.UN": [ "TSE"], # !
         },

    "oil_gas": 
        {
            "ARX" : ["TSE"],  # ++++++++++
            "SU" :  ["TSE"], # ++++++ # "NYSE"
            "TRP" : ["TSE"], # ++++++ # "NYSE"
            "VET" : ["TSE"], # "NYSE"
            "OVV" : ["TSE"], # "NYSE"
            "CVE" : ["TSE"], # "NYSE"
            "CNQ" : ["TSE"], # "NYSE"
            "IPO" : ["TSE"],
            "RBY" : ["TSE"],
            "AKT.A": [ "TSE"], # !
            
        },

    "minerals":
        {
            "AFM": ["CDNX"], # !
            
            "MND": ["TSE"], # !
            "WM": ["TSE"], # !
            "PTM": ["TSE"], # !
            "LN": ["TSE"], # !
            
            "MSA" : ["TSE"], # !
            "ALV" : ["TSE"], # !
            "ORA" : ["TSE"], # !
        },
        
    "building":
        {
          "DRX": ["TSE"], # !
        },
       
       "entertainment":
           {
                "CJR.B": ["TSE"], # !
                "CGX": ["TSE"], # !
           },

    "apparel":
        {
          "GOOS": ["TSE"], # !
          "ATZ": ["TSE"], # !
          
        },
        
    "energy":
        {
            "EMA": ["TSE"], # !
            "E": ["TSE"], # !
            "DRX": ["TSE"], # !
            "KEI": ["TSE"], # !
            
        },

    "tech_software": 
        {
            "BB": ["TSE"], # !
            "DTOL": ["TSE"], # !
            "ECN": [ "TSE"], # !
            "WNDR": ["TSE"], # !
            "GRID": ["TSE"], # !
            
            "SHOP" : ["TSE"], # "NYSE"
            "OTEX" : ["TSE"], # "NYSE"
            "TRI" : ["TSE"] # "NYSE"
        },

    "tech_data_processing": 
        {
            "HUT": ["TSE"], # "NASDAQ"
            "BITF": ["TSE"] # "NASDAQ"
        },

    "transportation": 
        {
            "TFII" : ["TSE"], # "NYSE"
            "WCN" : ["TSE"], # "NYSE"
            "CP" : ["TSE"], # "NYSE"
        },

        "canadian_etf": 
        {
            "HYLD" : ["TSE"], # HAMILTON ENHANCED US COVE CALL ETF UNIT UNHEDGED CAD
            "USCL" : ["TSE"], # GLOBAL X ENHANCED S&P 500 COVERED C CL A UNIT
            "QQCL" : ["TSE"], # GLOBAL X ENHANCED NASDAQ 100 COVERE UNIT CL A
            "HDIV" : ["TSE"], # HAMILTON ENHANCED MLTI SCTR COVE CA EL E UNIT
            "HDIF" : ["TSE"], # HAMILTON ENHANCED DIVIDIDEND FINANCIALS ETF
            
            "BMAX" : ["TSE"], # BROMPTON ENHANCED MUL ASSET INC ETF UNIT
            
            "HUTS" : ["TSE"], # HORIZONS BLOCKCHAIN TECH ETF CLASS E UNITS  
            
            "XIC" : ["TSE"], # ISHARES CORE S&P/TSX CAPPED COMPOSITE INDEX ETF 
            "XIU" : ["TSE"], # ISHARES S&P/TSX 60 INDEX ETF
            "VFV" : ["TSE"], # VANGUARD S&P 500 INDEX ETF
            "ZSP" : ["TSE"], # BMO S&P 500 INDEX ETF          
            "ZCN" : ["TSE"], # BMO S&P/TSX CAPPED COMPOSITE IDX ETF
            
            "XEI" : ["TSE"], # ISHARES S&P/TSX COMPOSITE HIGH DIV UNITS
            
            "ZWB" : ["TSE"], # BMO COVERED CALL CANADIAN BANKS ETF UNIT
            "ZWC" : ["TSE"], # BMO CDN HIGH DIVID COVERED CALL ETF CAD UNIT
            "ZWU" : ["TSE"], # BMO COVERED CALL UTILITIES ETF UNIT
            "ZRP" : ["TSE"], # BMO LADDERED PREFERRED SHS IDX ETF UNITS ETF
            "HHL" : ["TSE"], # HARVEST HEALTHCARE LEADERS INCM ETF CLASS A UNITS
            "HMAX" : ["TSE"], # HAMILTON CDN FINANCIAL YLD MAXIMIZER ETF
            "HTA" : ["TSE"], # HARVEST TECH ACHIEVRS GWT & INM ETF CLASS A UNITS
            "TXF" : ["TSE"], # CI TECH GIANTS COVERED CALL ETF HEDGED COMMON UNITS
            "HPYT" : ["TSE"], # HARVEST PREMIUM YIELD TREASURY ETF UNIT CL A
            "UMAX" : ["TSE"], # HAMILTON UTILS YIELD MAXIMIZER ETF UNIT CL E
            "ZMMK" : ["TSE"], # BMO MONEY MKT FD ETF SER UNIT
            "XUU" : ["TSE"], 
            "XRE" : ["TSE"], 
            "XBB" : ["TSE"], 
            "XPF" : ["TSE"], 
            "XIT" : ["TSE"], 
            "XSP" : ["TSE"],
            
            "FIE" : ["TSE"], # ISHARES CDN FINANCIAL MTHLY INC FD COM UNIT
            
            "ZHY" : ["TSE"], # BMO HIGH YIELD US CORP BOND HEDGED TO CAD ETF # !
            "ZFH" : ["TSE"], # BMO FLOATING RATE HIGH YIELD ETF CAD UNITS
            

        },

        "canadian_leveraged_etf" : 
        {
            "HQU" : ["TSE"],

        },

    "usa_banks": 
        {
            "C" : ["NYSE"], 
            "BAC": ["NYSE"], 
            "JPM": ["NYSE"],
            "GS" : ["NYSE"],
            "MS" : ["NYSE"]
        },

    "usa": 
        {
            "AVGO" : ["NASDAQ"],
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

    "usa_cad_hadged":
    {

    },

    "etf": 
        {
            "SPY" : ["NYSE"], 
            "DIA" : ["NYSE"]
        },

    "leveraged_etf" : 
        {
            "TQQQ" : ["NASDAQ"], 
            "SQQQ" : ["NASDAQ"], 
            "QLD" : ["NYSE"]
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

def get_selected_tickers() -> Generator[tuple[str, list[str]], None, None]:
    g = [(key, ticker_groups[key]) for key in selected_ticker_group_names]
    for group, tickers in g:
       for ticker, exchanges in tickers.items():
           yield (ticker, exchanges)

def get_selected_tickers_list() -> list[tuple[str, list[str]]]:
    return list(get_selected_tickers())

def get_all_tickers_list() -> list[tuple[str, list[str]]]:
    return list(get_all_tickers())

def get_all_tickers() -> Generator[tuple[str, list[str]], None, None]:
    for values in ticker_groups.values():
        for key, value in values.items():
            yield (key, value)

def get_test_tickers() -> Generator[tuple[str, list[str]], None, None]:
    l = get_all_tickers_list()
    yield l[0]
    yield l[1]

def get_test_tickers_batches(batch_size:int) -> Generator[list[tuple[str, list[str]]], None, None]:
    return batch_generator(get_test_tickers(), batch_size)

def get_all_tickers_batches_list(batch_size:int) -> list[list[tuple[str, list[str]]]]:
    return list(get_all_tickers_batches(batch_size))

def get_all_tickers_batches(batch_size:int) -> Generator[list[tuple[str, list[str]]], None, None]:
    return batch_generator(get_all_tickers(), batch_size)

def get_selected_tickers_batches_list(batch_size:int) -> list[list[tuple[str, list[str]]]]:
    return list(get_selected_tickers_batches(batch_size))

def get_selected_tickers_batches(batch_size:int) -> Generator[list[tuple[str, list[str]]], None, None]:
    return batch_generator(get_selected_tickers(), batch_size)

def batch_generator(sequence:Generator[tuple[str, list[str]], None, None], batch_size:int) \
    -> Generator[list[tuple[str, list[str]]], None, None]:
    batch = []
    for item in sequence:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_even_items(list:list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]] :
    return list[::2]

def get_odd_items(list:list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]] :
    return list[1::2]
