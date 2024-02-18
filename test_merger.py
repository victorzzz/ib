import multiprocessing
import ib_raw_data_merger as ib_merger

import ib_tickers as ib_tckrs
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import constants as cnts

import ib_tickers as ib_tckrs

if __name__ == "__main__":

    raw_files:list[str] = list(fsu.iterate_files(cnts.data_folder))
    selected_tickets_batches_list: list[list[tuple[str, list[str]]]] = list(ib_tckrs.get_selected_tickers_batches(2))

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = (dict[str, int])(manager.dict())

    first: list[tuple[str, list[str]]] = selected_tickets_batches_list[0]
    ib_merger.merge_csv_files(first, raw_files, lock, shared_dict)