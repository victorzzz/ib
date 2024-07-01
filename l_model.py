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
                 dropout:float):
        super(TransformerEncoderModel, self).__init__()

        self.save_hyperparameters(ignore=["model"])
        
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingForEncoder(d_model, max_pos_encoder_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model*max_pos_encoder_length, out_dim)

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
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        
        """
        r2 = self.train_r2(y_hat, y)
        self.log('train_r2', r2, on_epoch=True)
        """
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        """
        r2 = self.train_r2(y_hat, y)
        self.log('val_r2', r2, on_epoch=True)
        """
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assuming the batch is a tuple (X, Y) and we need only X for prediction
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)