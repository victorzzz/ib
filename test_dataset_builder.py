import multiprocessing
import ib_dataset_builder as ib_dataset_builder

import ib_tickers as ib_tckrs
import file_system_utils as fsu
import df_date_time_utils as df_dt_utils
import constants as cnts

import ib_tickers as ib_tckrs

if __name__ == "__main__":

    fsu.create_required_folders()

    selected_tickets_batches_list: list[list[tuple[str, list[str]]]] = list(ib_tckrs.get_selected_tickers_batches(2))

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = (dict[str, int])(manager.dict())

    first: list[tuple[str, list[str]]] = selected_tickets_batches_list[0]
    ib_dataset_builder.create_datasets(first, lock, shared_dict)