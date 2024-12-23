from l_input_data_definition import *
import df_tech_indicator_utils as df_tiu

EMA_PERIODS_1m_SHORT_SEQ:list[int] = [2, 4, 8]
EMA_PERIODS_1m_LONG_SEQ:list[int] = [16, 32, 64, 128, 192, 256]
BB_PERIODS_1m_SEQ:list[int] = [128, 192, 256]

day_trading_prediction_distance_0:int = 64
day_trading_prediction_distance_1:int = day_trading_prediction_distance_0 + day_trading_prediction_distance_0 // 2
day_trading_prediction_distance_2:int = day_trading_prediction_distance_0 * 2

ti_1m_distance:int = day_trading_prediction_distance_0 // 8
category_1m_distance:int = day_trading_prediction_distance_0 // 16

EMA_PERIODS_10m_SHORT_SEQ:list[int] = [2, 4]
EMA_PERIODS_10m_LONG_SEQ:list[int] = [32, 64, 128, 192, 256]
BB_PERIODS_10m_SEQ:list[int] = [128, 192, 256]

ti_10m_distance:int = day_trading_prediction_distance_0 // 16

EMA_PERIODS_30m_SHORT_SEQ:list[int] = [2, 4]
EMA_PERIODS_30m_LONG_SEQ:list[int] = [64, 128, 192, 256]
BB_PERIODS_30m_SEQ:list[int] = [128, 192, 256]

ti_30m_distance:int = day_trading_prediction_distance_0 // 32

EMA_PERIODS_390m_SHORT_SEQ:list[int] = [2, 4]
EMA_PERIODS_390m_LONG_SEQ:list[int] = [128, 192, 256]
BB_PERIODS_390m_SEQ:list[int] = [128, 192, 256]

swing_prediction_distance_days:int = 5
swing_prediction_distance_0:int = swing_prediction_distance_days * 390
swing_prediction_distance_1:int = swing_prediction_distance_0 * 2
swing_prediction_distance_2:int = swing_prediction_distance_0 * 3

ti_390m_distance:int = swing_prediction_distance_days

sequences:SEQUENCES_TYPE = [
    
    ######################################################
    # 1m
    ######################################################

    (
        1,
        day_trading_prediction_distance_2,
        [DATA_EMA_RATIO],
        EMA_PERIODS_1m_SHORT_SEQ, 
        [
            '1m_TRADES_average'
        ]
    ),    
    (
        1,
        day_trading_prediction_distance_1,
        [DATA_BB_RATIO],
        BB_PERIODS_1m_SEQ, 
        [
            '1m_TRADES_average'
        ]
    ),   
    (
        1,
        day_trading_prediction_distance_0,
        [DATA_BASE_EMA_RATIO],
        EMA_PERIODS_1m_LONG_SEQ,
        [
            '1m_TRADES_average',
            
            '1m_BID_close', 
            '1m_ASK_close',
            
            '1m_BID_open', 
            '1m_ASK_open',            
            
            '1m_BID_low', 
            '1m_ASK_low',               
            
            '1m_BID_high', 
            '1m_ASK_high',               
            
            '1m_TRADES_open',
            '1m_TRADES_close',
            '1m_TRADES_high', 
            '1m_TRADES_low'
        ]
    ),    
    (
        1,
        ti_1m_distance,
        [DATA_VALUE],
        [],
        [            
            '1m__t_MFI_TRADES_average_7', '1m__t_MFI_TRADES_average_14', '1m__t_MFI_TRADES_average_21',
            '1m__t_CCI_TRADES_average_7', '1m__t_CCI_TRADES_average_14', '1m__t_CCI_TRADES_average_21',
            
            '1m__t_RSI_TRADES_average_7', '1m__t_RSI_TRADES_average_14', '1m__t_RSI_TRADES_average_21',
            
            '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3',
            '1m__t_STOCH_k_TRADES_average_21_4', '1m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
    (
        1,
        category_1m_distance,
        [DATA_CATEGORY],
        [],
        [
            "1m_day_of_week", 
            "1m_week_of_year", 
            "1m_in_10m",
            "1m_in_30m",
            "1m_day_15_minute_period",
            "1m_day_26_minute_period"            
        ]
    ),

    ######################################################
    # 10m
    ######################################################
    
    (
        10,
        day_trading_prediction_distance_2,
        [DATA_EMA_RATIO],
        EMA_PERIODS_10m_SHORT_SEQ, 
        [
            '10m_TRADES_average'
        ]
    ),    
    (
        10,
        day_trading_prediction_distance_1,
        [DATA_BB_RATIO],
        BB_PERIODS_10m_SEQ, 
        [
            '10m_TRADES_average'
        ]
    ),   
    (
        10,
        day_trading_prediction_distance_0,
        [DATA_BASE_EMA_RATIO],
        EMA_PERIODS_10m_LONG_SEQ,
        [
            '10m_TRADES_average',
            
            '10m_BID_close', 
            '10m_ASK_close',
            
            '10m_BID_open', 
            '10m_ASK_open',            
            
            '10m_BID_low', 
            '10m_ASK_low',               
            
            '10m_BID_high', 
            '10m_ASK_high',               
            
            '10m_TRADES_open',
            '10m_TRADES_close',
            '10m_TRADES_high', 
            '10m_TRADES_low'
        ]
    ),    
    (
        10,
        ti_10m_distance,
        [DATA_VALUE],
        [],
        [            
            '10m__t_MFI_TRADES_average_7', '10m__t_MFI_TRADES_average_14', '10m__t_MFI_TRADES_average_21',
            '10m__t_CCI_TRADES_average_7', '10m__t_CCI_TRADES_average_14', '10m__t_CCI_TRADES_average_21',
            
            '10m__t_RSI_TRADES_average_7', '10m__t_RSI_TRADES_average_14', '10m__t_RSI_TRADES_average_21',
            
            '10m__t_STOCH_k_TRADES_average_14_3', '10m__t_STOCH_d_TRADES_average_14_3',
            '10m__t_STOCH_k_TRADES_average_21_4', '10m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
      
    ######################################################
    # 30m
    ######################################################

    (
        30,
        day_trading_prediction_distance_2,
        [DATA_EMA_RATIO],
        EMA_PERIODS_30m_SHORT_SEQ, 
        [
            '30m_TRADES_average'
        ]
    ),    
    (
        30,
        day_trading_prediction_distance_1,
        [DATA_BB_RATIO],
        BB_PERIODS_30m_SEQ, 
        [
            '30m_TRADES_average'
        ]
    ),   
    (
        30,
        day_trading_prediction_distance_0,
        [DATA_BASE_EMA_RATIO],
        EMA_PERIODS_30m_LONG_SEQ,
        [
            '30m_TRADES_average',
            
            '30m_BID_close', 
            '30m_ASK_close',
            
            '30m_BID_open', 
            '30m_ASK_open',            
            
            '30m_BID_low', 
            '30m_ASK_low',               
            
            '30m_BID_high', 
            '30m_ASK_high',               
            
            '30m_TRADES_open',
            '30m_TRADES_close',
            '30m_TRADES_high', 
            '30m_TRADES_low'
        ]
    ),    
    (
        30,
        ti_30m_distance,
        [DATA_VALUE],
        [],
        [            
            '30m__t_MFI_TRADES_average_7', '30m__t_MFI_TRADES_average_14', '30m__t_MFI_TRADES_average_21',
            '30m__t_CCI_TRADES_average_7', '30m__t_CCI_TRADES_average_14', '30m__t_CCI_TRADES_average_21',
            
            '30m__t_RSI_TRADES_average_7', '30m__t_RSI_TRADES_average_14', '30m__t_RSI_TRADES_average_21',
            
            '30m__t_STOCH_k_TRADES_average_14_3', '30m__t_STOCH_d_TRADES_average_14_3',
            '30m__t_STOCH_k_TRADES_average_21_4', '30m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
   
    ######################################################
    # 390m - 1 day
    ######################################################    

    (
        390,
        swing_prediction_distance_days * 16,
        [DATA_EMA_RATIO],
        EMA_PERIODS_390m_SHORT_SEQ,
        [            
            '390m_TRADES_average'
        ]
    ), 
    (
        390,
        swing_prediction_distance_days * 8,
        [DATA_BB_RATIO],
        BB_PERIODS_390m_SEQ,
        [            
            '390m_TRADES_average'
        ]
    ),
    (
        390,
        swing_prediction_distance_days * 8,
        [DATA_BASE_EMA_RATIO],
        EMA_PERIODS_390m_LONG_SEQ,
        [
            '390m_TRADES_average',
            
            '390m_BID_close', 
            '390m_ASK_close',
            
            '390m_BID_open', 
            '390m_ASK_open',            
            
            '390m_BID_low', 
            '390m_ASK_low',               
            
            '390m_BID_high', 
            '390m_ASK_high',               
            
            '390m_TRADES_open',
            '390m_TRADES_close',
            '390m_TRADES_high', 
            '390m_TRADES_low'
        ]
    ),         
    (
        390,
        ti_390m_distance,
        [DATA_VALUE],
        [],
        [            
            '390m__t_MFI_TRADES_average_7', '390m__t_MFI_TRADES_average_14', '390m__t_MFI_TRADES_average_21',
            '390m__t_CCI_TRADES_average_7', '390m__t_CCI_TRADES_average_14', '390m__t_CCI_TRADES_average_21',
            
            '390m__t_RSI_TRADES_average_7', '390m__t_RSI_TRADES_average_14', '390m__t_RSI_TRADES_average_21',
            
            '390m__t_STOCH_k_TRADES_average_14_3', '390m__t_STOCH_d_TRADES_average_14_3',
            '390m__t_STOCH_k_TRADES_average_21_4', '390m__t_STOCH_d_TRADES_average_21_4',            
        ]  
    ),
]

pred_columns:PRED_COLUMNS_TYPE = [
    (1, day_trading_prediction_distance_0, '1m_TRADES_average', (PRED_MIN, PRED_MAX_BEFORE_MIN, PRED_MAX, PRED_MIN_BEFORE_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, day_trading_prediction_distance_1, '1m_TRADES_average', (PRED_MIN, PRED_MAX_BEFORE_MIN, PRED_MAX, PRED_MIN_BEFORE_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, day_trading_prediction_distance_2, '1m_TRADES_average', (PRED_MIN, PRED_MAX_BEFORE_MIN, PRED_MAX, PRED_MIN_BEFORE_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, swing_prediction_distance_0, '1m_TRADES_average', (PRED_MIN, PRED_MAX_BEFORE_MIN, PRED_MAX, PRED_MIN_BEFORE_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, swing_prediction_distance_1, '1m_TRADES_average', (PRED_MIN, PRED_MAX_BEFORE_MIN, PRED_MAX, PRED_MIN_BEFORE_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    ]

log_columns:LOG_COLUMNS_TYPE = []

scaling_column_groups:SCALING_COLUMN_GROUPS_TYPE = []
