import pandas as pd
import numpy as np
import logging
import constants as cnts
import multiprocessing
import ib_logging as ib_log

def procEntryPoint():
    ib_log.configure_logging()

    for i in range(100):
        logging.info(f"Test-info, {i}")
        logging.error(f"Test-error, {i}")

if __name__ == "__main__":
    ib_log.configure_logging()

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

    # Example DataFrame
    data = {
        'AAA_one': [1, 2, 3],
        'AAA_two': [4, 5, 6],
        'BBB_three': [7, 8, 9],
        'CCC_four': [10, 11, 12]
    }
    df = pd.DataFrame(data)

    # Get list of columns that do not start with 'AAA'
    columns_except_AAA = [col for col in df.columns if not str(col).startswith('AAA')]

    print(columns_except_AAA)

    logging.error(f"Test {'abc'}")

    process1 = multiprocessing.Process(target=procEntryPoint, args=())
    process1.start()
    
    process2 = multiprocessing.Process(target=procEntryPoint, args=())
    process2.start()
    
    process1.join()
    process2.join()



