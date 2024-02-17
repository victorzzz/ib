import ib_raw_data_merger as ib_merger

import ib_tickers as ib_tckrs
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import constants as cnts

import ib_tickers as ib_tckrs

raw_files:list[str] = list(fsu.iterate_files(cnts.data_folder))

all_tickets_batches_list: list[list[tuple[str, dict[str, int]]]] = list(ib_tckrs.get_all_tickets_batches(2)) 

first: list[tuple[str, dict[str, int]]] = all_tickets_batches_list[0]
ib_merger.merge_csv_files(first, raw_files)