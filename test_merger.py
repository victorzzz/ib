import ib_raw_data_merger as ib_merger
from typing import Tuple
from typing import Dict
from typing import List
import ib_tickers as ib_tckrs
import file_systen_utils as fsu
import df_date_time_utils as df_dt_utils
import constants as cnts
from typing import List, Tuple, Dict
import ib_tickers as ib_tckrs

raw_files:List[str] = list(fsu.iterate_files(cnts.data_folder))

all_tickets_batches_list: List[List[Tuple[str, Dict[str, int]]]] = list(ib_tckrs.get_all_tickets_batches(2)) 

first: List[Tuple[str, Dict[str, int]]] = all_tickets_batches_list[0]
ib_merger.merge_csv_files(first, raw_files)