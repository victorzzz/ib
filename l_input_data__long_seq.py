from l_input_data_definition import *
import df_tech_indicator_utils as df_tiu

EMA_PERIODS_LONG_SEQ:list[int] = [16, 32, 64, 128, 192, 256]
EMA_PERIODS_DAY_LONG_SEQ:list[int] = [16, 32, 64, 128, 192]

EMA_PERIODS_SHORT_SEQ:list[int] = [4, 8, 12]
EMA_PERIODS_SHORT_SEQ_VOLUME:list[int] = [2, 4]

prediction_distance:int = 8
long_prediction_distance_days:int = 5
long_prediction_distance:int = long_prediction_distance_days * 390

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

pred_columns:PRED_COLUMNS_TYPE = [
    (1, prediction_distance, '1m_ASK_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, prediction_distance, '1m_BID_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER),
    (1, long_prediction_distance, '1m_ASK_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER), 
    (1, long_prediction_distance, '1m_BID_close', (PRED_MIN, PRED_MAX), (PRED_TRANSFORM_RATIO,), df_tiu.PRICE_RATIO_MULTIPLIER),    
    ]

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
