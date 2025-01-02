
DATA_CATEGORY:str = "category"
DATA_VALUE:str = "value"

DATA_EMA:str = "ema"
DATA_EMA_DIFF:str = "diff_ema"
DATA_EMA_RATIO:str = "ratio_ema"

DATA_LOG_EMA_RATIO:str = "ratio_log_ema"

DATA_BASE_EMA_RATIO:str = "ratio_base_ema"

DATA_BB1_LOW:str = "bb1_low"
DATA_BB1_LOW_DIFF:str = "diff_bb1_low"
DATA_BB1_LOW_RATIO:str = "ratio_bb1_low"

DATA_BB1_MID:str = "bb1_mid"
DATA_BB1_MID_DIFF:str = "diff_bb1_mid"
DATA_BB1_MID_RATIO:str = "ratio_bb1_mid"

DATA_BB1_HI:str = "bb1_hi"
DATA_BB1_HI_DIFF:str = "diff_bb1_hi"
DATA_BB1_HI_RATIO:str = "ratio_bb1_hi"

DATA_BB2_LOW:str = "bb2_low"
DATA_BB2_LOW_DIFF:str = "diff_bb2_low"
DATA_BB2_LOW_RATIO:str = "ratio_bb2_low"

DATA_BB2_MID:str = "bb2_mid"
DATA_BB2_MID_DIFF:str = "diff_bb2_mid"
DATA_BB2_MID_RATIO:str = "ratio_bb2_mid"

DATA_BB2_HI:str = "bb2_hi"
DATA_BB2_HI_DIFF:str = "diff_bb2_hi"
DATA_BB2_HI_RATIO:str = "ratio_bb2_hi"

DATA_MIN:str = "min"
DATA_MIN_DIFF:str = "diff_min"
DATA_MIN_RATIO:str = "ratio_min"

DATA_MAX:str = "max"
DATA_MAX_DIFF:str = "diff_max"
DATA_MAX_RATIO:str = "ratio_max"

DATA_VP_64:str = "vp_64"
DATA_VP_128:str = "vp_128"
DATA_VP_192:str = "vp_192"
DATA_VP_256:str = "vp_256"
DATA_VP_384:str = "vp_384"

PRED_MIN:str = "pred_min"
PRED_MAX:str = "pred_max"
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
