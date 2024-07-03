import torch
import torch.nn as nn
import torchmetrics

import lightning as L

import numpy as np

# Positional Encoding for Transformer
class PositionalEncodingForEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncodingForEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class TransformerEncoderModel(L.LightningModule):
    def __init__(self, 
                 input_dim:int, 
                 d_model:int, 
                 out_dim:int, 
                 max_pos_encoder_length:int, 
                 nhead:int, 
                 num_layers:int,
                 encoder_dim_feedforward:int,
                 num_decoder_layers:int,
                 use_banchnorm_for_decoder:bool,
                 use_dropout_for_decoder:bool, 
                 dropout:float,
                 dropout_for_decoder:float,
                 learning_rate:float=0.001):
        super(TransformerEncoderModel, self).__init__()

        self.save_hyperparameters(ignore=["model"])
        
        self.learning_rate = learning_rate
        
        self.train_r2 = torchmetrics.R2Score(num_outputs=out_dim)
        self.val_r2 = torchmetrics.R2Score(num_outputs=out_dim)
        
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_smape = torchmetrics.SymmetricMeanAbsolutePercentageError ()
        self.val_rse = torchmetrics.RelativeSquaredError(num_outputs=out_dim)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.max_pos_encoder_length = max_pos_encoder_length
        self.out_dim = out_dim
        self.num_decoder_layers = num_decoder_layers
        self.use_banchnorm_for_decoder = use_banchnorm_for_decoder
        self.use_dropout_for_decoder = use_dropout_for_decoder
        self.dropout = dropout
        self.dropout_for_decoder = dropout_for_decoder

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingForEncoder(d_model, max_pos_encoder_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=self.encoder_dim_feedforward, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder =  self.generate_decoder()

    def generate_decoder(self) -> nn.Module:
        
        decoder = torch.nn.Sequential()
        in_dim = self.d_model * self.max_pos_encoder_length
        
        for _ in range(0, self.num_decoder_layers):
            out_dim = in_dim // 4
            
            decoder.append(torch.nn.Linear(in_dim, out_dim))
            if self.use_banchnorm_for_decoder:
                decoder.append(torch.nn.BatchNorm1d(out_dim))
            decoder.append(torch.nn.ReLU())
            if self.use_dropout_for_decoder:
                decoder.append(torch.nn.Dropout(p=self.dropout_for_decoder))

            in_dim //= 4
        
        decoder.append(torch.nn.Linear(in_dim , self.out_dim))
        
        return decoder
        

    def forward(self, x):
        
        # x shape: [batch_size, seq_len, input_dim]
        
        batch_size, _, _ = x.size()
        
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Flatten the output from (batch_size, seq_length, d_model) to (batch_size, seq_length * d_model)
        x = x.view(batch_size, -1)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) # Flatten the target tensor. Ensure y has shape [batch_size, out_dim]
        
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        
        r2 = self.train_r2(y_hat, y)
        self.log('train_r2', r2, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) # Flatten the target tensor. Ensure y has shape [batch_size, out_dim]

        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        r2 = self.train_r2(y_hat, y)
        self.log('val_r2', r2, on_epoch=True)
        
        mape = self.val_mape(y_hat, y)
        self.log('val_mape', mape, on_epoch=True, prog_bar=True)
        
        smape = self.val_smape(y_hat, y)
        self.log('val_smape', smape, on_epoch=True)
        
        rse = self.val_rse(y_hat, y)
        self.log('val_rse', rse, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assuming the batch is a tuple (X, Y) and we need only X for prediction
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        """
        opt = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate,
            momentum=0.9)
        return opt
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)