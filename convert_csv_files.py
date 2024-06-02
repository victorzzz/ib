import constants as cnts
import file_system_utils as fs_utils
import logging
import ib_logging as ib_log
import ib_tickers_cache as ib_tickers_cache

import df_loader_saver as df_ls

ib_log.configure_logging("convert_csv_files")

logging.info(f"Starting {__file__} ...")

fs_utils.create_required_folders()

df_ls.convert_files_in_folder("test_data", "csv", "parquet")

df_ls.convert_files_in_folder(cnts.data_folder, "csv", "parquet")
df_ls.convert_files_in_folder(cnts.data_archived_folder, "csv", "parquet")
df_ls.convert_files_in_folder(cnts.merged_data_folder, "csv", "parquet")
df_ls.convert_files_in_folder(cnts.data_sets_folder, "csv", "parquet")

