import torch
import torch.nn as nn
import lightning as L
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(L.LightningModule):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, seq_len=256, pred_len=8, step=2):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.num_predictions = pred_len  # Number of future predictions based on the step

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, self.num_predictions * 2)  # Output dimension: pred_len * 2 (high and low)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Generate the mask once during initialization
        self.tgt_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(
            src, tgt,
            tgt_mask=self.tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output[:, -1, :])  # Use the last time step output for prediction
        output = output.view(-1, self.num_predictions, 2)  # Reshape to (batch_size, num_predictions, num_features)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt, y = batch
        src_key_padding_mask = (src[:, :, 0] == 0)
        tgt_key_padding_mask = (tgt[:, :, 0] == 0)
        output = self(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = nn.MSELoss()(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
