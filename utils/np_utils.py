import numpy as np

def custom_shift(arr, shift, fill_value=np.nan):
    if shift == 0:
        return arr
    elif shift > 0:
        return np.concatenate([np.full(shift, fill_value), arr[:-shift]])
    else:
        return np.concatenate([arr[-shift:], np.full(-shift, fill_value)])
