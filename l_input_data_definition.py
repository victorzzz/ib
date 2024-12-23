
DATA_CATEGORY:str = "category"
DATA_VALUE:str = "value"

DATA_EMA:str = "ema"
DATA_EMA_DIFF:str = "diff_ema"
DATA_EMA_RATIO:str = "ratio_ema"
DATA_BASE_EMA_RATIO:str = "ratio_base_ema"


DATA_BB:str = "bb"
DATA_BB_DIFF:str = "diff_bb"
DATA_BB_RATIO:str = "ratio_bb"

PRED_MIN:str = "pred_min"
PRED_MAX_BEFORE_MIN:str = "pred_max_before_min"

PRED_MAX:str = "pred_max"
PRED_MIN_BEFORE_MAX:str = "pred_min_before_max"

PRED_AVG:str = "pred_avg"
PRED_FIRST:str = "pred_first"
PRED_LAST:str = "pred_last"
PRED_LAST_OBSERVED:str = "pred_last_observed"

PRED_TRANSFORM_NONE:str = "pred_transform_none"
PRED_TRANSFORM_DIFF:str = "pred_transform_diff"
PRED_TRANSFORM_RATIO:str = "pred_transform_ratio"

time_ranges:list[int] = [1, 3, 10, 30, 390]

# each tuple: (candle sticks time range, prediction_distance, data_types, ema_periods, data_columns)
SEQUENCES_TYPE = list[tuple[int, int, list[str], list[int], list[str]]]

# each tuple: (candle sticks time range, prediction_distance, column_name, prediction type, prediction transform, price ratio multiplier)
PRED_COLUMNS_TYPE = list[tuple[int, int, str, tuple[str, ...], tuple[str, ...], float]]

# each tuple: (candle sticks time range, column_name, is time range based scaled down)
LOG_COLUMNS_TYPE = list[tuple[int, str, bool]]

SCALING_COLUMN_GROUP_METADATA_TYPE = tuple[int, str, bool]
SCALING_COLUMN_GROUP_CONTENT_TYPE = list[tuple[int, list[str]]]
SCALING_COLUMN_GROUP_TYPE = tuple[SCALING_COLUMN_GROUP_METADATA_TYPE, SCALING_COLUMN_GROUP_CONTENT_TYPE]
SCALING_COLUMN_GROUPS_TYPE = list[SCALING_COLUMN_GROUP_TYPE]
