import l_model as lm

sequences:list[tuple[int,list[str]]] = [
    (
        256 + 128, 
        [
            '1m_BID_close', '1m_ASK_close',
            '1m_BID_high', '1m_BID_low',
            '1m_ASK_high', '1m_ASK_low',
            '1m_BID_open', '1m_ASK_open',
            '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            '1m_MIDPOINT_close',
            '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            #'1m_TRADES_volume', 
            '1m_TRADES_average',
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
            '1m_MIDPOINT_close', 
            '1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            '1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            '1m_TRADES_average'
        ],
        False),
    
    # '1m_TRADES_volume': ([], True)   
    }

prediction_distance:int = 16
d_model_param:int = 32
nhead_param:int = 32
num_layers_param:int = 6
encoder_dim_feedforward_param:int = 1024
num_decoder_layers_param:int = 6
use_banchnorm_for_decoder_param:bool = True
use_dropout_for_decoder_param:bool = True
dropout_param:float = 0.1
dropout_for_decoder:float = 0.2
learning_rate_param:float = 0.0001

batch_size_param:int = 32

def create_model() -> lm.TransformerEncoderModel:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2
    
    model = lm.TransformerEncoderModel(
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
        
        learning_rate=learning_rate_param
    )
    
    return model    

def load_model(path:str) -> lm.TransformerEncoderModel:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns) * 2

    model = lm.TransformerEncoderModel.load_from_checkpoint(
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
        learning_rate=learning_rate_param
    )
    
    return model