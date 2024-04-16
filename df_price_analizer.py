import numpy as np

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