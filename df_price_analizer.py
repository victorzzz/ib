import numpy as np
import pandas as pd

PRICE_FIELD:str = 'TRADES_average'
DAY_FIELD:str = 'normalized_day_of_week'

DEFAULT_TARGET_PROFITS_PERCENTS:list[float] = [0.9, 1.5, 2.1]

DEFAULT_PROFIT_LOSS_RATIO:float = 3.0

def getArrayPairs(arr:np.ndarray) -> np.ndarray:
    pairs = np.column_stack((np.roll(arr, 1), arr))[1:]
    return pairs

def getCheckPointsIntervals(chek_points:np.ndarray) -> np.ndarray:
    # get indexes of chek-points
    check_points_indexes = np.where(chek_points == 1)[0]

    # get indexes of intervals between chekpoints
    intervals_between_chekpoints = getArrayPairs(check_points_indexes)

    return intervals_between_chekpoints

# returns (increases, decreases)
def calculateDayPriceChangeWithTrailingStopLoss(
        prices:np.ndarray,
        days:np.ndarray,
        trailingStopLosses:np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    
    # asserts
    assert len(prices) == len(days)

    l_prices = len(prices)
    l_losses = len(trailingStopLosses)

    day_borders = np.where( (np.diff(days, append=np.nan) != 0), 1, 0)

    day_borders_intervals = getCheckPointsIntervals(day_borders)
    if (len(day_borders_intervals) > 0):
        first_interval = day_borders_intervals[0]
        first_interval_begin = first_interval[0]
        day_borders_intervals = np.vstack((np.array([-1, first_interval_begin]), day_borders_intervals))   

    inc_results_for_losses = np.zeros((l_losses, l_prices))
    dec_results_for_losses = np.zeros((l_losses, l_prices))

    for interval in day_borders_intervals:
        begin = interval[0]
        end = interval[1]

        price_end = prices[end]
        previous = price_end

        inc_current_max_arr = np.full_like(trailingStopLosses, price_end)
        inc_current_min = price_end

        dec_current_min_arr = np.full_like(trailingStopLosses, price_end)
        dec_current_max = price_end

        for i in range(end-1, begin, -1):
            i_price = prices[i]

            # inc
            inc_current_max_greater_price = inc_current_max_arr >= i_price
            inc_results_for_losses[inc_current_max_greater_price, i] = (inc_current_max_arr[inc_current_max_greater_price] - i_price) / i_price

            inc_current_max_less_price = inc_current_max_arr < i_price
            inc_current_max_arr[inc_current_max_less_price] = i_price

            # dec
            dec_current_min_less_price = dec_current_min_arr <= i_price
            dec_results_for_losses[dec_current_min_less_price, i] = (i_price - dec_current_min_arr[dec_current_min_less_price]) / i_price

            dec_current_min_greater_price = dec_current_min_arr > i_price
            dec_current_min_arr[dec_current_min_greater_price] = i_price

            # inc
            if (i_price > previous):
                    inc_dif_to_min = (i_price - inc_current_min) / i_price

                    inc_stopLoss_less_diff_to_min =  trailingStopLosses < inc_dif_to_min
                    inc_results_for_losses[inc_stopLoss_less_diff_to_min, i] = 0.0
                    inc_current_max_arr[inc_stopLoss_less_diff_to_min] = i_price

            if (i_price <= inc_current_min):
                inc_current_min = i_price 

            # dec
            if (i_price < previous):
                    dec_dif_to_max = (dec_current_max - i_price) / i_price

                    dec_stopLoss_less_diff_to_max =  trailingStopLosses < dec_dif_to_max
                    dec_results_for_losses[dec_stopLoss_less_diff_to_max, i] = 0.0
                    dec_current_min_arr[dec_stopLoss_less_diff_to_max] = i_price

            if (i_price >= dec_current_max):
                dec_current_max = i_price             

            previous = i_price

    return (inc_results_for_losses, dec_results_for_losses)

# add new columns to the dataframe
#   - for each target profit add new column of type int with long position label 1 if profit is reached and 0 otherwise
#   - for each target profit add new column of type int with short position label 1 if profit is reached and 0 otherwise
# returns new dataframe
def addPriceChangeLabelsToDataFrame(
    df:pd.DataFrame, 
    target_profits_percents:list[float] = DEFAULT_TARGET_PROFITS_PERCENTS, 
    profit_lost_ratio:float = DEFAULT_PROFIT_LOSS_RATIO ) -> pd.DataFrame:

    prices:np.ndarray = df[PRICE_FIELD].to_numpy(dtype=np.float32)
    days:np.ndarray = df[DAY_FIELD].to_numpy(dtype=np.float32)

    l_profits:int = len(target_profits_percents)

    profits:np.ndarray = np.array(target_profits_percents, dtype=np.float32) / 100.0
    trailing_stop_losses:np.ndarray = profits / profit_lost_ratio

    inc_results_for_losses, dec_results_for_losses = calculateDayPriceChangeWithTrailingStopLoss(prices, days, trailing_stop_losses)
    
    for i, profit in enumerate(target_profits_percents):
        df[f"long_profit_{profit}".replace(".", "_")] = np.where(inc_results_for_losses[i] >= profits[i], 1, 0)
        df[f"long_exit_{profit}".replace(".", "_")] = np.where(dec_results_for_losses[i] >= trailing_stop_losses[i], 1, 0)
        
        df[f"short_profit_{profit}".replace(".", "_")] = np.where(dec_results_for_losses[i] >= profits[i], 1, 0)
        df[f"short_exit_{profit}".replace(".", "_")] = np.where(inc_results_for_losses[i] >= trailing_stop_losses[i], 1, 0)
        
        df = df.copy()
        
    return df