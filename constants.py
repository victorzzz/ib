import numpy as np
from typing import List

data_folder = "IB_Data"
data_archived_folder = "IB_Data_Archived"
merged_data_folder = "IB_MergedData"
error_investigation_folder = "IB_ErrorInvestigation"
tickers_cache_folder = "IB_TickersCache"

stock_events_folder = "IB_StockEvents"
financials_folder = "IB_Financials"
dividends_folder = "IB_Dividends"

test_data_folder = "TestData"
test_merged_data_folder = "TestMergedData"

data_sets_folder = "ib_data_sets"
data_sets_summary_folder = "ib_data_sets_summary"

data_long_term_sets_folder = "ib_data_long_term_sets"
data_long_term_sets_summary_folder = "ib_data_long_term_sets_summary"

data_sets_for_training_folder = "ib_data_sets_for_training"

merged_data_with_indicators_folder = "ib_merged_data_with_indicators"
merged_data_with_vp_folder = "ib_merged_data_with_vp"

second_multipliers = {
    1.0: "1 secs", 
    5.0: "5 secs", 
    10.0: "10 secs", 
    15.0: "15 secs", 
    30.0: "30 secs" 
}

minute_multipliers = {
    1.0: "1 min", 
    # 2.0: "2 mins", 
    3.0: "3 mins", 
    # 5.0: "5 mins", 
    10.0: "10 mins", 
    30.0: "30 mins",
    #60.0: "1 hour",
    120.0: "2 hours",
    390.0: "1 day",
    }

long_term_buy_thresholds = (0.021, 0.035, 0.049)

loss_levels = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0])
gain_levels = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0])

loss_level_for_training = 0.625
gain_level_for_training = 1.125


complex_processing_batch_size = 3

float_nan = float("nan")
