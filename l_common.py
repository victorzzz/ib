import l_model as lm

sequences:list[tuple[int,list[str]]] = [
    (
        256, 
        [
            #'1m_BID_close', '1m_ASK_close',
            #'1m_BID_high', '1m_BID_low',
            #'1m_ASK_high', '1m_ASK_low',
            #'1m_BID_open', '1m_ASK_open',
            #'1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            '1m_MIDPOINT_close',
            #'1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            '1m_TRADES_volume', '1m_TRADES_average'
        ]
    )]

pred_columns:list[str] = ['1m_MIDPOINT_close']

scaling_column_groups:dict[str, tuple[list[str], bool]] = {
    '1m_MIDPOINT_close': 
        ([
            #'1m_BID_close', '1m_ASK_close',
            #'1m_BID_high', '1m_BID_low',
            #'1m_ASK_high', '1m_ASK_low',
            #'1m_BID_open', '1m_ASK_open',
            #'1m_MIDPOINT_open', '1m_MIDPOINT_high', '1m_MIDPOINT_low', 
            #'1m_TRADES_open', '1m_TRADES_high', '1m_TRADES_low', '1m_TRADES_close',
            '1m_TRADES_average'
        ],
        False),
    
    '1m_TRADES_volume': ([], True)   
    }

prediction_distance:int = 8
d_model_param:int = 8
nhead_param:int = 4
num_layers_param:int = 3
dropout_param:float = 0.1

def create_model() -> lm.TransformerEncoderModel:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)
    
    model = lm.TransformerEncoderModel(
        input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        dropout=dropout_param
    )
    
    return model    

def load_model(path:str) -> lm.TransformerEncoderModel:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)

    model = lm.TransformerEncoderModel.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        dropout=dropout_param
    )
    
    return model