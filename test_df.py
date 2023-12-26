import pandas as pd
import numpy as np

# Creating DataFrame A
data_A = {'I': [1, 2, 3], 'F1': [11, 22, 33]}
df_A = pd.DataFrame(data_A)

# Creating DataFrame B
data_B = {'I': [1, 2, 4], 'F2': [55, 66, 77]}
df_B = pd.DataFrame(data_B)

# Setting 'I' as the index for both DataFrames
df_A.set_index('I', inplace=True)
df_B.set_index('I', inplace=True)

# Merging the DataFrames horizontally
result = pd.concat([df_A, df_B], axis=1, sort=True)

# Displaying the result
print(result)
