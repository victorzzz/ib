import l_model as lm
import l_ff_model as l_ff_m
import l_module as lmodule
import df_tech_indicator_utils as df_tiu

EMA_PERIODS_LONG_SEQ:list[int] = [16, 32, 64, 128, 192, 256]
EMA_PERIODS_DAY_LONG_SEQ:list[int] = [16, 32, 64, 128, 192]

EMA_PERIODS_SHORT_SEQ:list[int] = [4, 8, 12]
EMA_PERIODS_SHORT_SEQ_VOLUME:list[int] = [2, 4]

DATA_CATEGORY:str = "category"
DATA_VALUE:str = "value"
DATA_EMA:str = "ema"
DATA_EMA_DIFF:str = "diff_ema"
DATA_EMA_RATIO:str = "ratio_ema"

PRED_MIN:str = "pred_min"
PRED_MAX:str = "pred_max"
PRED_AVG:str = "pred_avg"
PRED_FIRST:str = "pred_first"
PRED_LAST:str = "pred_last"
PRED_LAST_OBSERVED:str = "pred_last_observed"

# PRED_MINUS_1:str = "pred_minus_1"

PRED_TRANSFORM_NONE:str = "pred_transform_none"
PRED_TRANSFORM_DIFF:str = "pred_transform_diff"
PRED_TRANSFORM_RATIO:str = "pred_transform_ratio"

prediction_distance:int = 8
long_prediction_distance_days:int = 5
long_prediction_distance:int = long_prediction_distance_days * 390

time_ranges:list[int] = [1, 3, 10, 30, 390]

# each tuple: (candle sticks time range, prediction_distance, data_types, ema_periods, data_columns)
SEQUENCES_TYPE = list[tuple[int, int, list[str], list[int], list[str]]]
sequences:SEQUENCES_TYPE = [
    
    ######################################################
    # 1m
    ######################################################

    (
        1,
        prediction_distance * 24,
        [DATA_VALUE],
        EMA_PERIODS_LONG_SEQ, 
        [
            '1m_TRADES_average'
        ]
    ),    
    (
        1,
        prediction_distance * 16,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_LONG_SEQ, 
        [
            '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_average',
            '1m_BID_high', '1m_BID_low', '1m_ASK_high', '1m_ASK_low', 
        ]
    ),   
    (
        1,
        prediction_distance * 8,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ,
        [
            '1m_BID_close', 
            '1m_ASK_close',
            '1m_BID_high', 
            '1m_BID_low',
            '1m_ASK_high', 
            '1m_ASK_low',
            '1m_BID_open', 
            '1m_ASK_open',
            
            '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            '1m_MIDPOINT_close',
            
            '1m_TRADES_open', '1m_TRADES_close',
            '1m_TRADES_high', '1m_TRADES_low',
            '1m_TRADES_average'
        ]
    ),
    (
        1,
        prediction_distance,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ_VOLUME,
        [            
            '1m_TRADES_volume_LOG', '1m_TRADES_barCount_LOG',
            '1m_TRADES_open', '1m_TRADES_close',
            '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_average' 
        ]
    ),    
    (
        1,
        prediction_distance // 2,
        [DATA_VALUE],
        [],
        [      
            '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_average',      
            
            '1m__t_MFI_TRADES_average_7', '1m__t_MFI_TRADES_average_14', '1m__t_MFI_TRADES_average_21',
            '1m__t_CCI_TRADES_average_7', '1m__t_CCI_TRADES_average_14', '1m__t_CCI_TRADES_average_21',
            
            '1m__t_RSI_TRADES_average_7', '1m__t_RSI_TRADES_average_14', '1m__t_RSI_TRADES_average_21',
            
            '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20',             
            '1m__t_BBL_TRADES_average_30', '1m__t_BBM_TRADES_average_30', '1m__t_BBU_TRADES_average_30', 
            
            '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3',
            '1m__t_STOCH_k_TRADES_average_21_4', '1m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
    (
        1,
        prediction_distance // 2,
        [DATA_CATEGORY],
        [],
        [
            "1m_day_of_week", 
            "1m_week_of_year", 
            "1m_trading_10_minute_period",
            "1m_trading_15_minute_period",
            "1m_trading_26_minute_period"            
        ]
    ),
    
    ######################################################
    # 3m
    ######################################################
    
    (
        3,
        prediction_distance * 24,
        [DATA_VALUE],
        EMA_PERIODS_LONG_SEQ, 
        [
            '3m_TRADES_average'
        ]
    ),    
    (
        3,
        prediction_distance * 16,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_LONG_SEQ, 
        [
            '3m_TRADES_high', '3m_TRADES_low', '3m_TRADES_average',
            '3m_BID_high', '3m_BID_low', '3m_ASK_high', '3m_ASK_low', 
        ]
    ),
    (
        3,
        prediction_distance * 8,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ,
        [
            '3m_BID_close', 
            '3m_ASK_close',
            
            '3m_BID_open',  
            '3m_ASK_open',            
            
            '3m_MIDPOINT_open', '3m_MIDPOINT_high', '3m_MIDPOINT_low', 
            '3m_MIDPOINT_close',
            
            '3m_TRADES_open', '3m_TRADES_close',
            '3m_TRADES_high', '3m_TRADES_low',
            '3m_TRADES_average'
        ]
    ),
    (
        3,
        prediction_distance,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ_VOLUME,
        [
            '3m_TRADES_volume_LOG', '3m_TRADES_barCount_LOG',
            '3m_TRADES_average'            
        ]
    ),    
    (
        3,
        prediction_distance // 2,
        [DATA_VALUE],
        [],
        [    
            '3m_TRADES_high', '3m_TRADES_low', '3m_TRADES_average',      
                             
            '3m__t_MFI_TRADES_average_7', '3m__t_MFI_TRADES_average_14', '3m__t_MFI_TRADES_average_21',
            '3m__t_CCI_TRADES_average_7', '3m__t_CCI_TRADES_average_14', '3m__t_CCI_TRADES_average_21',
            '3m__t_RSI_TRADES_average_7', '3m__t_RSI_TRADES_average_14', '3m__t_RSI_TRADES_average_21',
            
            '3m__t_BBL_TRADES_average_20', '3m__t_BBM_TRADES_average_20', '3m__t_BBU_TRADES_average_20', 
            '3m__t_BBL_TRADES_average_30', '3m__t_BBM_TRADES_average_30', '3m__t_BBU_TRADES_average_30', 
            
            '3m__t_STOCH_k_TRADES_average_14_3', '3m__t_STOCH_d_TRADES_average_14_3',
            '3m__t_STOCH_k_TRADES_average_21_4', '3m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
    
    ######################################################
    # 10m
    ######################################################
    
    (
        10,
        prediction_distance * 16,
        [DATA_VALUE],
        EMA_PERIODS_LONG_SEQ, 
        [
            '10m_TRADES_average'
        ]
    ),    
    (
        10,
        prediction_distance * 12,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_LONG_SEQ, 
        [
            '10m_TRADES_high', '10m_TRADES_low', '10m_TRADES_average',
            '10m_BID_high', '10m_BID_low', '10m_ASK_high', '10m_ASK_low',            
        ]
    ),
    (
        10,
        prediction_distance * 6,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ,
        [
            '10m_BID_close', 
            '10m_ASK_close',
            
            '10m_BID_open',  
            '10m_ASK_open',
            
            '10m_MIDPOINT_open', '10m_MIDPOINT_high', '10m_MIDPOINT_low', 
            '10m_MIDPOINT_close',
            
            '10m_TRADES_open', '10m_TRADES_close',
            '10m_TRADES_high', '10m_TRADES_low',
            '10m_TRADES_average'
        ]
    ),
    (
        10,
        prediction_distance // 2,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ_VOLUME,
        [
            '10m_TRADES_volume_LOG', '10m_TRADES_barCount_LOG',
            '10m_TRADES_average'
        ]
    ),    
    (
        10,
        prediction_distance // 4,
        [DATA_VALUE],
        [],
        [         
            '10m_TRADES_high', '10m_TRADES_low', '10m_TRADES_average',           
            
            '10m__t_MFI_TRADES_average_7', '10m__t_MFI_TRADES_average_14', '10m__t_MFI_TRADES_average_21',
            '10m__t_CCI_TRADES_average_7', '10m__t_CCI_TRADES_average_14', '10m__t_CCI_TRADES_average_21',
            '10m__t_RSI_TRADES_average_7', '10m__t_RSI_TRADES_average_14', '10m__t_RSI_TRADES_average_21',
            
            '10m__t_BBL_TRADES_average_20', '10m__t_BBM_TRADES_average_20', '10m__t_BBU_TRADES_average_20', 
            '10m__t_BBL_TRADES_average_30', '10m__t_BBM_TRADES_average_30', '10m__t_BBU_TRADES_average_30', 
            
            '10m__t_STOCH_k_TRADES_average_14_3', '10m__t_STOCH_d_TRADES_average_14_3',
            '10m__t_STOCH_k_TRADES_average_21_4', '10m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),  
    
    ######################################################
    # 30m
    ######################################################
    
    (
        30,
        prediction_distance * 12,
        [DATA_VALUE],
        EMA_PERIODS_LONG_SEQ, 
        [
            '30m_TRADES_average'
        ]
    ),    
    (
        30,
        prediction_distance * 8,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_LONG_SEQ, 
        [
            '30m_TRADES_high', '30m_TRADES_low', '30m_TRADES_average'
        ]
    ),
    (
        30,
        prediction_distance * 4,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ,
        [
            '30m_BID_close', 
            '30m_ASK_close',
            
            '30m_BID_open',  
            '30m_ASK_open',
            
            '30m_MIDPOINT_open', '30m_MIDPOINT_high', '30m_MIDPOINT_low', 
            '30m_MIDPOINT_close',
            
            '30m_TRADES_open', '30m_TRADES_close',
            '30m_TRADES_high', '30m_TRADES_low',
            '30m_TRADES_average'
        ]
    ),
    (
        30,
        prediction_distance // 2,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ_VOLUME,
        [
            '30m_TRADES_volume_LOG', '30m_TRADES_barCount_LOG',
            '30m_TRADES_average'
        ]
    ),    
    (
        30,
        prediction_distance // 4,
        [DATA_VALUE],
        [],
        [
            '30m_TRADES_high', '30m_TRADES_low', '30m_TRADES_average',              
                        
            '30m__t_MFI_TRADES_average_7', '30m__t_MFI_TRADES_average_14', '30m__t_MFI_TRADES_average_21',
            '30m__t_CCI_TRADES_average_7', '30m__t_CCI_TRADES_average_14', '30m__t_CCI_TRADES_average_21',
            '30m__t_RSI_TRADES_average_7', '30m__t_RSI_TRADES_average_14', '30m__t_RSI_TRADES_average_21',
            
            '30m__t_BBL_TRADES_average_20', '30m__t_BBM_TRADES_average_20', '30m__t_BBU_TRADES_average_20', 
            '30m__t_BBL_TRADES_average_30', '30m__t_BBM_TRADES_average_30', '30m__t_BBU_TRADES_average_30', 
            
            '30m__t_STOCH_k_TRADES_average_14_3', '30m__t_STOCH_d_TRADES_average_14_3',
            '30m__t_STOCH_k_TRADES_average_21_4', '30m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
    
    ######################################################
    # 390m - 1 day
    ######################################################    
    
    (
        390,
        long_prediction_distance_days * 16,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_DAY_LONG_SEQ, 
        [
            '390m_TRADES_high', '390m_TRADES_low', '390m_TRADES_average',        
        ]
    ),
    (
        390,
        long_prediction_distance_days * 8,
        [DATA_VALUE, DATA_EMA],
        EMA_PERIODS_SHORT_SEQ, 
        [      
            '390m_TRADES_open', '390m_TRADES_close',
            '390m_TRADES_high', '390m_TRADES_low', '390m_TRADES_average',   
            '390m_BID_open', '390m_ASK_open',
            '390m_BID_close', '390m_ASK_close',
            '390m_BID_high', '390m_BID_low', '390m_ASK_high', '390m_ASK_low', 
        ]
    ),    
    (
        390,
        long_prediction_distance_days,
        [DATA_VALUE, DATA_EMA_RATIO],
        EMA_PERIODS_SHORT_SEQ_VOLUME,
        [
            '390m_TRADES_volume_LOG', '390m_TRADES_barCount_LOG',
            '390m_TRADES_average',             
        ]
    ),     
    (
        390,
        long_prediction_distance_days,
        [DATA_VALUE],
        [],
        [
            '390m_TRADES_high', '390m_TRADES_low', '390m_TRADES_average',               
                        
            '390m__t_MFI_TRADES_average_7', '390m__t_MFI_TRADES_average_14', '390m__t_MFI_TRADES_average_21',
            '390m__t_CCI_TRADES_average_7', '390m__t_CCI_TRADES_average_14', '390m__t_CCI_TRADES_average_21',
            '390m__t_RSI_TRADES_average_7', '390m__t_RSI_TRADES_average_14', '390m__t_RSI_TRADES_average_21',
            
            '390m__t_BBL_TRADES_average_20', '390m__t_BBM_TRADES_average_20', '390m__t_BBU_TRADES_average_20', 
            '390m__t_BBL_TRADES_average_30', '390m__t_BBM_TRADES_average_30', '390m__t_BBU_TRADES_average_30', 
            
            '390m__t_STOCH_k_TRADES_average_14_3', '390m__t_STOCH_d_TRADES_average_14_3',
            '390m__t_STOCH_k_TRADES_average_21_4', '390m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),                      
]

# each tuple: (candle sticks time range, prediction_distance, column_name, prediction type, prediction transform, price ratio multiplier)
PRED_COLUMNS_TYPE = list[tuple[int, int, str, tuple[str, ...], tuple[str, ...], float]]
pred_columns:PRED_COLUMNS_TYPE = [
    (1, prediction_distance, '1m_ASK_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, prediction_distance, '1m_BID_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER),
    (1, long_prediction_distance, '1m_ASK_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, long_prediction_distance, '1m_BID_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER),    
    ]

# each tuple: (candle sticks time range, column_name, is time range based scaled down)
LOG_COLUMNS_TYPE = list[tuple[int, str, bool]]
log_columns:LOG_COLUMNS_TYPE = [
    (1, '1m_TRADES_volume', False),    
    (3, '3m_TRADES_volume', True),
    (10, '10m_TRADES_volume', True),
    (30, '30m_TRADES_volume', True),
    (390, '390m_TRADES_volume', True),
    
    (1, '1m_TRADES_barCount', False),    
    (3, '3m_TRADES_barCount', True),
    (10, '10m_TRADES_barCount', True),
    (30, '30m_TRADES_barCount', True),    
    (390, '390m_TRADES_barCount', True),    
]

SCALING_COLUMN_GROUP_METADATA_TYPE = tuple[int, str, bool]
SCALING_COLUMN_GROUP_CONTENT_TYPE = list[tuple[int, list[str]]]
SCALING_COLUMN_GROUP_TYPE = tuple[SCALING_COLUMN_GROUP_METADATA_TYPE, SCALING_COLUMN_GROUP_CONTENT_TYPE]
SCALING_COLUMN_GROUPS_TYPE = list[SCALING_COLUMN_GROUP_TYPE]
scaling_column_groups:SCALING_COLUMN_GROUPS_TYPE = [
    (
        (1, '1m_ASK_close', True), # default scaling group for prices 
        [
            (1,
            [
                # 1m
                ##################
                
                '1m_BID_close',
                '1m_BID_high', 
                '1m_BID_low',
                '1m_ASK_high', 
                '1m_ASK_low',
                '1m_BID_open', 
                '1m_ASK_open',
                
                '1m_MIDPOINT_close', 
                '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
                
                '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
                '1m_TRADES_average',
                
                '1m__t_BBL_TRADES_average_30', '1m__t_BBM_TRADES_average_30', '1m__t_BBU_TRADES_average_30', 
                '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20', 
            ]),
            (3,
            [    
                # 3m
                ##################
                
                '3m_ASK_close',
                '3m_BID_close',
                
                '3m_ASK_open',
                '3m_BID_open',                
                
                '3m_BID_high', 
                '3m_BID_low',
                '3m_ASK_high', 
                '3m_ASK_low',
                
                '3m_MIDPOINT_close', 
                '3m_MIDPOINT_open', '3m_MIDPOINT_high', '3m_MIDPOINT_low', 
                
                '3m_TRADES_open', '3m_TRADES_high', '3m_TRADES_low', '3m_TRADES_close',
                '3m_TRADES_average',
                
                '3m__t_BBL_TRADES_average_30', '3m__t_BBM_TRADES_average_30', '3m__t_BBU_TRADES_average_30', 
                '3m__t_BBL_TRADES_average_20', '3m__t_BBM_TRADES_average_20', '3m__t_BBU_TRADES_average_20', 
            ]),
            (10,
            [
                # 10m
                ##################
                
                '10m_ASK_close',
                '10m_BID_close',
                
                '10m_ASK_open',
                '10m_BID_open',                
                
                '10m_BID_high', 
                '10m_BID_low',
                '10m_ASK_high', 
                '10m_ASK_low',
                
                '10m_MIDPOINT_close', 
                '10m_MIDPOINT_open', '10m_MIDPOINT_high', '10m_MIDPOINT_low', 
                
                '10m_TRADES_open', '10m_TRADES_high', '10m_TRADES_low', '10m_TRADES_close',
                '10m_TRADES_average',
                
                '10m__t_BBL_TRADES_average_30', '10m__t_BBM_TRADES_average_30', '10m__t_BBU_TRADES_average_30', 
                '10m__t_BBL_TRADES_average_20', '10m__t_BBM_TRADES_average_20', '10m__t_BBU_TRADES_average_20', 
            ]),
            (30,
            [
                # 30m
                ##################
                
                '30m_ASK_close',
                '30m_BID_close',
                
                '30m_ASK_open',
                '30m_BID_open',
                
                '30m_BID_high', 
                '30m_BID_low',
                '30m_ASK_high', 
                '30m_ASK_low',
                
                '30m_MIDPOINT_close', 
                '30m_MIDPOINT_open', '30m_MIDPOINT_high', '30m_MIDPOINT_low', 
                
                '30m_TRADES_open', '30m_TRADES_high', '30m_TRADES_low', '30m_TRADES_close',
                '30m_TRADES_average',
                
                '30m__t_BBL_TRADES_average_30', '30m__t_BBM_TRADES_average_30', '30m__t_BBU_TRADES_average_30', 
                '30m__t_BBL_TRADES_average_20', '30m__t_BBM_TRADES_average_20', '30m__t_BBU_TRADES_average_20',                             
            ]),
            (390,
            [
                # 390m - 1 day
                ##################
                
                '390m_ASK_close',
                '390m_BID_close',
                
                '390m_ASK_open',
                '390m_BID_open',
                
                '390m_BID_high', 
                '390m_BID_low',
                '390m_ASK_high', 
                '390m_ASK_low',                
                
                '390m_MIDPOINT_close', 
                '390m_MIDPOINT_open', '390m_MIDPOINT_high', '390m_MIDPOINT_low', 
                
                '390m_TRADES_open', '390m_TRADES_high', '390m_TRADES_low', '390m_TRADES_close',
                '390m_TRADES_average',
                
                '390m__t_BBL_TRADES_average_30', '390m__t_BBM_TRADES_average_30', '390m__t_BBU_TRADES_average_30', 
                '390m__t_BBL_TRADES_average_20', '390m__t_BBM_TRADES_average_20', '390m__t_BBU_TRADES_average_20',                             
            ])            
        ]
    ),
    (
        (1, '1m_TRADES_volume_LOG', False),
        [
            (1,
             [                
             ]),
            (3,
             [
                # 3m
                ##################
                
                '3m_TRADES_volume_LOG',
             ]),
            (10,
             [
                # 10m
                ##################
                
                '10m_TRADES_volume_LOG',
             ]),
            (30,
             [
                # 30m
                ##################
                
                '30m_TRADES_volume_LOG',
             ]),
            (390,
             [
                # 390m - 1 day
                ##################
                
                '390m_TRADES_volume_LOG',
             ])            
        ]
    ),
    (
        (1, '1m_TRADES_barCount_LOG', False),
        [
            (1,
             [                
             ]),
            (3,
             [
                # 3m
                ##################
                
                '3m_TRADES_barCount_LOG',
             ]),
            (10,
             [
                # 10m
                ##################
                
                '10m_TRADES_barCount_LOG',
             ]),
            (30,
             [
                # 30m
                ##################
                
                '30m_TRADES_barCount_LOG',
             ]),
            (390,
             [
                # 390m - 1 day
                ##################
                
                '390m_TRADES_barCount_LOG',
             ])            
        ]
    )    
]

dataset_tail:float = 0.2
    
max_epochs_param:int = 15

batch_size_param:int = 2048

d_model_param:int = 64
use_linear_embeding_layer_param:bool = True
num_embeding_linear_layers_param:int = 1
nhead_param:int = d_model_param // 64
num_layers_param:int = 6
encoder_dim_feedforward_param:int = d_model_param * 4
num_decoder_layers_param:int = 12
use_decoder_normalization_param:bool = True
use_banchnorm_for_decoder_param:bool = False
use_dropout_for_decoder_param:bool = True
dropout_for_embeding_param:float = 0.1
dropout_param:float = 0.1
dropout_for_decoder:float = 0.1
first_decoder_denominator:int = 1
next_decoder_denominator:int = 1
n_inputs_for_decoder_param:int = prediction_distance * 2

ff_denominators_param:tuple[int, int, int, int]=(4, 1, 1, 2)
ff_mum_layers_param:int = 2
ff_dropout_param:float = 0.2
ff_use_batchnorm_param:bool = True

use_transformer = False

learning_rate_param:float | None = 0.0002
# recomended value for Noam LR Scheduler for Transformers is 4000 in the paper Attention is All You Need 
# but this model is small so we can use a smaller value
scheduler_warmup_steps_param:int = 1000 
model_size_for_noam_scheduler_formula_param = 8192

def create_module() -> lmodule.TimeSeriesModule:

    if use_transformer:
        return create_l_module()
    else:
        return create_l_ff_module()

def load_module(path:str) -> lmodule.TimeSeriesModule:

    if use_transformer:
        return load_l_module(path)
    else:
        return load_l_ff_module(path)

def create_l_module() -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = lm.TransformerEncoderModel(
        input_dim=input_dim,  # Number of input features
        use_linear_embeding_layer=use_linear_embeding_layer_param,
        num_embeding_linear_layers=num_embeding_linear_layers_param,
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        encoder_dim_feedforward=encoder_dim_feedforward_param,
        num_decoder_layers=num_decoder_layers_param,
        use_decoder_normalization=use_decoder_normalization_param,
        use_banchnorm_for_decoder=use_banchnorm_for_decoder_param,
        use_dropout_for_decoder=use_dropout_for_decoder_param,
        dropout_for_embeding=dropout_for_embeding_param,
        dropout=dropout_param,
        dropout_for_decoder=dropout_for_decoder,
        first_decoder_denominator=first_decoder_denominator,
        next_decoder_denominator=next_decoder_denominator,
        n_inputs_for_decoder=n_inputs_for_decoder_param)
    
    module = lmodule.TimeSeriesModule(
        model,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param)
    
    return module    

def load_l_module(path:str) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = lm.TransformerEncoderModel(
        input_dim=input_dim,  # Number of input features
        use_linear_embeding_layer=use_linear_embeding_layer_param,
        num_embeding_linear_layers=num_embeding_linear_layers_param,
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        encoder_dim_feedforward=encoder_dim_feedforward_param,
        num_decoder_layers=num_decoder_layers_param,
        use_decoder_normalization=use_decoder_normalization_param,
        use_banchnorm_for_decoder=use_banchnorm_for_decoder_param,
        use_dropout_for_decoder=use_dropout_for_decoder_param,
        dropout_for_embeding=dropout_for_embeding_param,
        dropout=dropout_param,
        dropout_for_decoder=dropout_for_decoder,
        first_decoder_denominator=first_decoder_denominator,
        next_decoder_denominator=next_decoder_denominator,
        n_inputs_for_decoder=n_inputs_for_decoder_param)
    
    module = lmodule.TimeSeriesModule.load_from_checkpoint(
        path,
        model=model,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param )
    
    return module

def create_l_ff_module() -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, seq_len, _, _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = l_ff_m.FFModel(
        input_dim=input_dim,  # Number of input features
        seq_len=max_seq_len,
        out_dim=out_dim,  # Number of output features
        num_layers=ff_mum_layers_param,
        denominators=ff_denominators_param,
        dropout=ff_dropout_param,
        ff_use_batchnorm=ff_use_batchnorm_param
    )
    
    module = lmodule.TimeSeriesModule(
        model,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param)
    
    return module    

def load_l_ff_module(path:str) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = l_ff_m.FFModel(
        input_dim=input_dim,  # Number of input features
        seq_len=max_seq_len,
        out_dim=out_dim,  # Number of output features
        num_layers=ff_mum_layers_param,
        denominators=ff_denominators_param,
        dropout=ff_dropout_param,
        ff_use_batchnorm=ff_use_batchnorm_param)
    
    module = lmodule.TimeSeriesModule.load_from_checkpoint(
        path,
        model=model,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param )
    
    return module

if __name__ == "__main__":
    pass
