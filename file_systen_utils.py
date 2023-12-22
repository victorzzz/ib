import os
import datetime

def is_file_exists(file_path:str) -> bool:
    return os.path.isfile(file_path)

def get_file_creation_datetime(file_path:str) -> datetime:
    # Get file metadata
    file_time = os.path.getmtime(file_path)
    
    # Get creation time
    creation_time = datetime.datetime.fromtimestamp(file_time)

    return creation_time
    
def iterate_files(folder_path:str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path

def iterate_files_name_only(folder_path:str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            yield file  