import torch
import torch.nn as nn
import numpy as np

# Positional Encoding for Transformer
class PositionalEncodingForEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncodingForEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderModel(nn.Module):
    
    def __init__(self, 
                 input_dim:int, 
                 use_linear_embeding_layer:bool,
                 num_embeding_linear_layers:int,
                 d_model:int, 
                 out_dim:int, 
                 max_pos_encoder_length:int, 
                 nhead:int, 
                 num_layers:int,
                 encoder_dim_feedforward:int,
                 num_decoder_layers:int,
                 use_decoder_normalization:bool,
                 use_banchnorm_for_decoder:bool,
                 use_dropout_for_decoder:bool,
                 dropout_for_embeding:float, 
                 dropout:float,
                 dropout_for_decoder:float,
                 first_decoder_denominator:int,
                 next_decoder_denominator:int,
                 n_inputs_for_decoder:int):
        
        super(TransformerEncoderModel, self).__init__()
        
        self.input_dim = input_dim
        self.use_linear_embeding_layer = use_linear_embeding_layer
        self.num_embeding_linear_layers = num_embeding_linear_layers
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.max_pos_encoder_length = max_pos_encoder_length
        self.out_dim = out_dim
        self.num_decoder_layers = num_decoder_layers
        self.use_decoder_normalization = use_decoder_normalization
        self.use_banchnorm_for_decoder = use_banchnorm_for_decoder
        self.use_dropout_for_decoder = use_dropout_for_decoder
        self.dropout_for_embeding = dropout_for_embeding
        self.dropout = dropout
        self.dropout_for_decoder = dropout_for_decoder
        self.first_decoder_denominator = first_decoder_denominator
        self.next_decoder_denominator = next_decoder_denominator
        self.n_inputs_for_decoder = n_inputs_for_decoder

        if not self.use_linear_embeding_layer:
            if self.d_model != self.input_dim:
                raise ValueError("d_model must be equal to input_dim when use_linear_embeding_layer is False")

        self.embeding = self.create_embeding()
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=self.encoder_dim_feedforward, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        nn.Transformer.generate_square_subsequent_mask(max_pos_encoder_length)
        
        self.decoder =  self.create_decoder()

    def forward(self, x):
        
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, _, _ = x.size()
        
        x = self.embeding(x)
        
        mask = nn.Transformer.generate_square_subsequent_mask(self.max_pos_encoder_length, device=x.device, dtype=x.dtype)
        x = self.transformer_encoder(x, mask=mask)
        
        # Flatten the output from (batch_size, seq_length, d_model) to (batch_size, seq_length * d_model)
        x = x[:, -self.n_inputs_for_decoder:, :].view(batch_size, -1)
        x = self.decoder(x)
        return x

    def create_embeding(self) -> nn.Module:
        embeding = torch.nn.Sequential()
        
        if self.use_linear_embeding_layer:
            embeding.append(torch.nn.Linear(self.input_dim, self.d_model))
            for i in range(0, self.num_embeding_linear_layers):
                embeding.append(torch.nn.LayerNorm(self.d_model))
                embeding.append(torch.nn.ReLU())
                embeding.append(torch.nn.Linear(self.d_model, self.d_model))

        embeding.append(PositionalEncodingForEncoder(self.d_model, self.max_pos_encoder_length))
        embeding.append(torch.nn.Dropout(p=self.dropout_for_embeding))
        
        return embeding

    def create_decoder(self) -> nn.Module:
            
            decoder = torch.nn.Sequential()
            
            in_dim = self.d_model * self.n_inputs_for_decoder
            
            out_dim = in_dim // self.first_decoder_denominator
            
            decoder.append(torch.nn.Linear(in_dim, out_dim))
            if self.use_decoder_normalization:
                if self.use_banchnorm_for_decoder:  
                    decoder.append(torch.nn.BatchNorm1d(out_dim))                  
                else:
                    decoder.append(torch.nn.LayerNorm(out_dim))  
            decoder.append(torch.nn.ReLU())
            if self.use_dropout_for_decoder:
                decoder.append(torch.nn.Dropout(p=self.dropout_for_decoder))
                    
            in_dim = out_dim
    
            for _ in range(0, self.num_decoder_layers):
                out_dim = in_dim // self.next_decoder_denominator
                
                decoder.append(torch.nn.Linear(in_dim, out_dim))
                if self.use_decoder_normalization:
                    if self.use_banchnorm_for_decoder:  
                        decoder.append(torch.nn.BatchNorm1d(out_dim))                  
                    else:
                        decoder.append(torch.nn.LayerNorm(out_dim)) 
                decoder.append(torch.nn.ReLU())
                if self.use_dropout_for_decoder:
                    decoder.append(torch.nn.Dropout(p=self.dropout_for_decoder))            

                in_dim = out_dim
            
            decoder.append(torch.nn.Linear(in_dim , self.out_dim))
            
            return decoder   
    