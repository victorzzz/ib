import csv
import pandas as pd

def get_headers(file:str) -> list[str]:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []

# returns a tuple with the headers and the first row    
def get_headers_with_first_row(file:str) -> tuple[list[str], list[str]]:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        
        try:
            headers = next(reader)
        except StopIteration:
            return [], []
        
        try:
            first_row = next(reader)
        except StopIteration:
            first_row = []

        return headers, first_row

def get_dataframe_first_row_only(file:str) -> pd.DataFrame:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        
        try:
            headers = next(reader)
        except StopIteration:
            return pd.DataFrame()
        
        try:
            first_row = next(reader)
        except StopIteration:
            first_row = []

        return pd.DataFrame([first_row], columns=headers)