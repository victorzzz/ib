import constants as cnts
import file_system_utils as fsu
import logging
import ib_logging as ib_log
import ib_tickers_cache as ib_tickers_cache
import df_loader_saver as df_ls
import df_volume_profile_utils as df_vp_utils

def add_valume_profiles_to_datasets(minute_multiplier:int):
    logging.info(f"Adding volume profiles to {minute_multiplier}-minute datasets ...")
    
    files = fsu.iterate_files(cnts.data_sets_folder)
    
    for file in files:
        if not f"{minute_multiplier}--minute--dataset" in file:
            logging.info(f"Skipping '{file}' ...")
            continue
        
        logging.info(f"Processing '{file}' ...")
        df = df_ls.load_df(file, format = None)
        
        df = df_vp_utils.add_top_of_volume_profile(df, price_column=f"{minute_multiplier}m_TRADES_average", volume_column=f"{minute_multiplier}m_TRADES_volume")
        
        df_ls.save_df(df, file)
        
        logging.info(f"Volume profiles added to '{file}'")


def do_step():
    add_valume_profiles_to_datasets(1)
    
# ----------------------------

if __name__ == "__main__":
    
    ib_log.configure_logging("ib_volume_profile_builder")

    logging.info(f"Starting {__file__} ...")

    do_step()