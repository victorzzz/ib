import logging

logs_folder = "Logs"
log_file = "Logs/log.txt"
error_log_file = "Logs/error_log.txt"

def configure_logging():
    # Configure logging
    # logging.basicConfig(filename=cnts.error_log_file, filemode="a", level=logging.ERROR, force=True, format='%(asctime)s| %(message)s')

    root = logging.getLogger()
    all_logs_file_handler = logging.FileHandler(log_file, 'a')
    all_logs_file_handler.setLevel(logging.DEBUG)

    warning_logs_file_handler = logging.FileHandler(error_log_file, 'a')
    warning_logs_file_handler.setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s| %(processName)-12s| %(name)-18s| %(levelname)-8s| %(message)s')

    all_logs_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    warning_logs_file_handler.setFormatter(formatter)

    root.addHandler(all_logs_file_handler)
    root.addHandler(warning_logs_file_handler)
    root.addHandler(console_handler)
    
    root.setLevel(logging.DEBUG)

    ib_insync_client_logger = logging.getLogger('ib_insync.client')
    ib_insync_client_logger.setLevel(logging.WARNING)

    ib_insync_wrapper_logger = logging.getLogger('ib_insync.wrapper')
    ib_insync_wrapper_logger.setLevel(logging.WARNING)