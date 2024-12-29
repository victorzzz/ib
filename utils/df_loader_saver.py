import pandas as pd
import pyarrow.parquet as pq
import os
import logging

def is_df_exists(file_pathwithout_extension:str, format:str = "parquet") -> bool:
    if format == "csv":
        return os.path.isfile(file_pathwithout_extension + ".csv")
    elif format == "parquet":
        return os.path.isfile(file_pathwithout_extension + ".parquet")
    else:
        raise ValueError("Invalid format")

def adjust_types32(df:pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].dtype == "int64":
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "float64":
            df[column] = df[column].astype("float32")

    return df

def load_df_first_timestamp(file_pathwithout_extension:str, format:str = "parquet") -> int | None:
    df:pd.DataFrame = load_df(file_pathwithout_extension, format, first_row_only=True, columns=["timestamp"])
    if df.empty or ("timestamp" not in df.columns):
        return None
    
    return df["timestamp"].iloc[0]

def load_df(file_path:str, format:str | None = "parquet", first_row_only:bool = False, columns:list[str] | None = None, adjust_types:bool = True) -> pd.DataFrame:
    result = load_df_internal(file_path, format, first_row_only, columns)
    if adjust_types:
        result = adjust_types32(result)
    
    return result

def load_df_internal(file_path:str, format:str | None = "parquet", first_row_only:bool = False, columns:list[str] | None = None, adjust_types:bool = True) -> pd.DataFrame:
    if format == "csv":
        if first_row_only:
            return pd.read_csv(file_path + ".csv", nrows=1, usecols=columns)
        else:
            return pd.read_csv(file_path + ".csv", usecols=columns)
    
    elif format == "parquet":
        if first_row_only:
            return load_parquet_first_group(file_path, columns=columns)
        else:
            return pd.read_parquet(file_path + ".parquet", engine="pyarrow", columns=columns)            
    
    elif format is None:
        filename, extension = os.path.splitext(file_path)
        
        if extension == ".csv":
            return load_df(filename, "csv", first_row_only=first_row_only, columns=columns)
        
        elif extension == ".parquet":
            return load_df(filename, "parquet", first_row_only=first_row_only, columns=columns)
        
        else:
            raise ValueError("Invalid extension")
            
    else:
        raise ValueError("Invalid format")

def load_parquet_first_group(file_pathwithout_extension:str, columns:list[str] | None) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(file_pathwithout_extension + ".parquet")
    first_row_group = parquet_file.read_row_group(0, use_threads=True, columns=columns)
    return first_row_group.to_pandas()

def save_df(df:pd.DataFrame, file_pathwithout_extension:str, format:str = "parquet", adjust_types:bool = True) -> None:
    if adjust_types:
        df = adjust_types32(df)
    
    save_df_internal(df, file_pathwithout_extension, format)

def save_df_internal(df:pd.DataFrame, file_pathwithout_extension:str, format:str = "parquet") -> None:
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