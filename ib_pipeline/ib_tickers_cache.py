import constants as cnts
import json
from ..utils import file_system_utils as fsu

import logging

tickers_cache_file:str = f"{cnts.tickers_cache_folder}/ib_tickers_cache.json"

def read_cached_data_if_needed(shared_tickers_cache:dict[str, int]) -> None:

    logging.debug(f"Reading cached data if needed")

    if len(shared_tickers_cache) == 0:

        logging.debug(f"Reading cached data from file")

        if fsu.is_file_exists(tickers_cache_file):
            with open(tickers_cache_file, "r") as f:
                shared_tickers_cache.update(json.load(f))

            logging.debug(f"Loaded tickers_cache: {shared_tickers_cache}")
        else:
            logging.debug(f"NO CAHE FILE YET. tickers_cache_file: {tickers_cache_file}")
            
    else:
        logging.debug(f"Cahce data is already in memory")
        logging.debug(f"CACHE IN MEMORY STATE. tickers_cache: {shared_tickers_cache}")

def get_contact_id(symbol:str, exchange:str, lock, shared_tickers_cache:dict[str, int]) -> int | None:
    
    logging.debug(f"Reading cached data for {symbol} {exchange}")

    with lock:

        logging.debug(f"Lock aquared. Reading cached data for {symbol} {exchange}")

        read_cached_data_if_needed(shared_tickers_cache)

        cache_key:str = f"{symbol}_{exchange}"

        if cache_key in shared_tickers_cache:
            return shared_tickers_cache[cache_key]
        else:
            return None
    
def add_contact_id(symbol:str, exchange:str, contact_id:int, lock, shared_tickers_cache:dict[str, int]) -> None:

    logging.debug(f"Adding cached data for {symbol} {exchange}")

    with lock:

        logging.debug(f"Lock aquared. Adding cached data for {symbol} {exchange} with contact_id {contact_id}")

        read_cached_data_if_needed(shared_tickers_cache)

        cache_key:str = f"{symbol}_{exchange}"

        shared_tickers_cache[cache_key] = contact_id

        logging.debug(f"Writing cached data to file for {symbol} {exchange} with contact_id {contact_id}")
        json_str = json.dumps(dict(shared_tickers_cache))
        logging.debug(f"json_str: {json_str}")

        with open(tickers_cache_file, "w") as f:
            f.write(json_str)

        logging.debug(f"DONE. Writing cached data to file for {symbol} {exchange} with contact_id {contact_id}")
