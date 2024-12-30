import l_model as lm
import l_ff_model as l_ff_m
import l_module as lmodule
from training import l_input_data_definition as l_input_data_def
from training import l_input_data__short_seq as l_input_data

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
n_inputs_for_decoder_param:int = l_input_data.day_trading_prediction_distance_0 * 2

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

def create_module(sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:

    if use_transformer:
        return create_l_module(sequences, pred_columns)
    else:
        return create_l_ff_module(sequences, pred_columns)

def load_module(path:str, sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:

    if use_transformer:
        return load_l_module(path, sequences, pred_columns)
    else:
        return load_l_ff_module(path, sequences, pred_columns)

def create_l_module(sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, _, _, _, columns in sequences)
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

def load_l_module(path:str, sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, _, _, _, columns in sequences)
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

def create_l_ff_module(sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, _, _, _, columns in sequences)
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

def load_l_ff_module(path:str, sequences:l_input_data_def.SEQUENCES_TYPE, pred_columns:l_input_data_def.PRED_COLUMNS_TYPE) -> lmodule.TimeSeriesModule:
    
    input_dim = sum(len(columns) for _, _, _, _, columns in sequences)
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
