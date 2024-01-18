import os
import shutil
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

def delete_file(file_path:str):
    if os.path.exists(file_path):
        os.remove(file_path)

def delete_folder(folder_path:str):
    if os.path.exists(folder_path):
        os.rmdir(folder_path)

def move_file_to_folder(file_path:str, destination_folder:str):
    """
    Moves a file from file_path to the destination_folder.

    :param file_path: The full path of the file to be moved.
    :param destination_folder: The folder where the file should be moved.
    """
    # Check if destination folder exists, create it if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move the file
    shutil.move(file_path, destination_folder)

def create_folder_if_not_exists(folder_path:str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)