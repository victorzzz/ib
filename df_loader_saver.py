from typing import Optional

import pandas as pd
import os
import logging

def is_df_exists(file_pathwithout_extension:str, format:str = "parquet") -> bool:
    if format == "csv":
        return os.path.isfile(file_pathwithout_extension + ".csv")
    elif format == "parquet":
        return os.path.isfile(file_pathwithout_extension + ".parquet")
    else:
        raise ValueError("Invalid format")

def load_df(file_pathwithout_extension:str, format:str = "parquet", nrows:Optional[int] = None) -> pd.DataFrame:
    if format == "csv":
        return pd.read_csv(file_pathwithout_extension + ".csv", nrows=nrows)
    elif format == "parquet":
        return pd.read_parquet(file_pathwithout_extension + ".parquet", engine="pyarrow", nrows=nrows)
    else:
        raise ValueError("Invalid format")

def save_df(df:pd.DataFrame, file_pathwithout_extension:str, format:str = "parquet") -> None:
    if format == "csv":
        df.to_csv(file_pathwithout_extension + ".csv", index=False)
    elif format == "parquet":
        df.to_parquet(file_pathwithout_extension + ".parquet", engine="pyarrow", index=False, compression="snappy")
    else:
        raise ValueError("Invalid format")

def convert_files_in_folder(folder:str, format:str, new_format:str, delete_original:bool = True) -> None:
    logging.info(f"Converting files in folder {folder} from {format} to {new_format}")

    if not os.path.isdir(folder):
        raise ValueError("Invalid folder")

    if format not in ["csv", "parquet"]:
        raise ValueError("Invalid format")
    
    if new_format not in ["csv", "parquet"]:
        raise ValueError("Invalid new_format")
    
    if format == new_format:
        raise ValueError("format and new_format must be different")

    for file in os.listdir(folder):
        if file.endswith("." + format):
            logging.info(f"Converting file {file}")
    
            file_pathwithout_extension = os.path.join(folder, os.path.splitext(file)[0])
            df = load_df(file_pathwithout_extension, format)
            save_df(df, file_pathwithout_extension, new_format)

            if delete_original:
                logging.info(f"Deleting file {file}")
                os.remove(file_pathwithout_extension + "." + format)
        else:
            logging.warning(f"Skipping file {file}")