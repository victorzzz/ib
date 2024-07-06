import l_model as lm

prediction_distance:int = 8

sequences:list[tuple[int,list[str]]] = [
    (
        prediction_distance * 16, 
        [
            '1m_BID_close', '1m_ASK_close',
            '1m_BID_high', '1m_BID_low',
            '1m_ASK_high', '1m_ASK_low',
            '1m_BID_open', '1m_ASK_open',
            #'1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            #'1m_MIDPOINT_close',
            '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            '1m_TRADES_average',
            
            '1m_TRADES_volume', 
            
            '1m__t_MFI_TRADES_average_7', '1m__t_MFI_TRADES_average_14', '1m__t_MFI_TRADES_average_21',
            '1m__t_CCI_TRADES_average_7', '1m__t_CCI_TRADES_average_14', '1m__t_CCI_TRADES_average_21',
            '1m__t_FI_TRADES_average_13', '1m__t_FI_TRADES_average_26',
            
            '1m__t_VPT_TRADES_average', 
            
            #'1m__t_NVI_TRADES_average',
            
            '1m__t_RSI_TRADES_average_7', '1m__t_RSI_TRADES_average_14', '1m__t_RSI_TRADES_average_21',
            
            '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20', 
            #'1m__t_BBP_TRADES_average_20', '1m__t_BBB_TRADES_average_20',
            
            '1m__t_BBL_TRADES_average_30', '1m__t_BBM_TRADES_average_30', '1m__t_BBU_TRADES_average_30', 
            #'1m__t_BBP_TRADES_average_30', '1m__t_BBB_TRADES_average_30',
            
            '1m__t_STOCH_k_TRADES_average_14_3', '1m__t_STOCH_d_TRADES_average_14_3',
            '1m__t_STOCH_k_TRADES_average_21_4', '1m__t_STOCH_d_TRADES_average_21_4',            
            
            
            # "1m_normalized_day_of_week", "1m_normalized_week", 
            "1m_normalized_trading_time"
        ]
    )]

pred_columns:list[str] = ['1m_BID_close', '1m_ASK_close']

scaling_column_groups:dict[str, tuple[list[str], bool]] = {
    '1m_BID_close': 
        ([
            # '1m_BID_close', 
            '1m_ASK_close',
            '1m_BID_high', '1m_BID_low',
            '1m_ASK_high', '1m_ASK_low',
            '1m_BID_open', '1m_ASK_open',
            
            # '1m_MIDPOINT_close', 
            # '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            
            '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            '1m_TRADES_average',
            
            '1m__t_BBL_TRADES_average_30', '1m__t_BBM_TRADES_average_30', '1m__t_BBU_TRADES_average_30', 
            '1m__t_BBL_TRADES_average_20', '1m__t_BBM_TRADES_average_20', '1m__t_BBU_TRADES_average_20', 
            
        ],
        False),
    
    '1m_TRADES_volume': ([], True)   
    }

dataset_tail:float = 0.03
    
max_epochs_param:int = 30

d_model_param:int = 32
nhead_param:int = 32
num_layers_param:int = 6
encoder_dim_feedforward_param:int = 1024
num_decoder_layers_param:int = 5
use_banchnorm_for_decoder_param:bool = True
use_dropout_for_decoder_param:bool = True
dropout_param:float = 0.1
dropout_for_decoder:float = 0.15
first_decoder_denominator:int = 2
next_decoder_denominator:int = 2

learning_rate_param:float | None = 0.0001
# recomended value for Noam LR Scheduler for Transformers is 4000 in the paper Attention is All You Need 
# but this model is small so we can use a smaller value
scheduler_warmup_steps_param:int = 1000 
model_size_for_noam_scheduler_formula_param = 8192

batch_size_param:int = 256 + 64

def create_model() -> lm.TransformerEncoderModule:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = lm.TransformerEncoderModule(
        input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        encoder_dim_feedforward=encoder_dim_feedforward_param,
        num_decoder_layers=num_decoder_layers_param,
        use_banchnorm_for_decoder=use_banchnorm_for_decoder_param,
        use_dropout_for_decoder=use_dropout_for_decoder_param,
        dropout=dropout_param,
        dropout_for_decoder=dropout_for_decoder,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        first_decoder_denominator=first_decoder_denominator,
        next_decoder_denominator=next_decoder_denominator,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param
    )
    
    return model    

def load_model(path:str) -> lm.TransformerEncoderModule:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2

    model = lm.TransformerEncoderModule.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        encoder_dim_feedforward=encoder_dim_feedforward_param,
        num_decoder_layers=num_decoder_layers_param,
        use_banchnorm_for_decoder=use_banchnorm_for_decoder_param,
        use_dropout_for_decoder=use_dropout_for_decoder_param,
        dropout=dropout_param,
        dropout_for_decoder=dropout_for_decoder,
        learning_rate=learning_rate_param,
        scheduler_warmup_steps=scheduler_warmup_steps_param,
        first_decoder_denominator=first_decoder_denominator,
        next_decoder_denominator=next_decoder_denominator,
        model_size_for_noam_scheduler_formula=model_size_for_noam_scheduler_formula_param   
    )
    
    return model