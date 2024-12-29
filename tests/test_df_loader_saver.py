import df_loader_saver as dls
import constants as cnts
import pandas as pd

# Path: test_df_loader_saver.py

df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
df["a"] = df["a"].astype("float16")

dls.save_df(df, "test_data/test_float")

"""
df_exists1 = dls.is_df_exists("test_data/FINAL_after_ASK_2020-12-08", "parquet")
df_exists2 = dls.is_df_exists(f"{cnts.merged_data_folder}/BMO-5094-NYSE--ib--1--minute--merged", "parquet")
df_exists3 = dls.is_df_exists(f"{cnts.data_folder}/CM-4458463--ib--1--minute--10--2007-08-21", "parquet")

print(df_exists1)
print(df_exists2)
print(df_exists3)
"""