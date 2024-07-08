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
                 next_decoder_denominator:int):
        
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
        x = x.view(batch_size, -1)
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
            
            in_dim = self.d_model * self.max_pos_encoder_length
            
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
    
class TransformerEncoderModule(L.LightningModule):
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
                 learning_rate:float | None,
                 scheduler_warmup_steps:int,
                 model_size_for_noam_scheduler_formula:int):
        super(TransformerEncoderModule, self).__init__()

        self.learning_rate = learning_rate
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.model_size_for_noam_scheduler_formula = model_size_for_noam_scheduler_formula
        
        self.model = TransformerEncoderModel(
            input_dim, 
            use_linear_embeding_layer,
            num_embeding_linear_layers,
            d_model, 
            out_dim, 
            max_pos_encoder_length, 
            nhead, num_layers, 
            encoder_dim_feedforward, 
            num_decoder_layers, 
            use_decoder_normalization,
            use_banchnorm_for_decoder, 
            use_dropout_for_decoder,
            dropout_for_embeding, 
            dropout, 
            dropout_for_decoder,
            first_decoder_denominator,
            next_decoder_denominator)

        self.save_hyperparameters()
    
        """
        self.train_r2 = torchmetrics.R2Score(num_outputs=out_dim)
        self.val_r2 = torchmetrics.R2Score(num_outputs=out_dim)
        
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_smape = torchmetrics.SymmetricMeanAbsolutePercentageError ()
        self.val_rse = torchmetrics.RelativeSquaredError(num_outputs=out_dim)
        """
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) # Flatten the target tensor. Ensure y has shape [batch_size, out_dim]
        
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) # Flatten the target tensor. Ensure y has shape [batch_size, out_dim]

        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        """
        r2 = self.train_r2(y_hat, y)
        self.log('val_r2', r2, on_epoch=True)
        
        mape = self.val_mape(y_hat, y)
        self.log('val_mape', mape, on_epoch=True, prog_bar=True)
        
        smape = self.val_smape(y_hat, y)
        self.log('val_smape', smape, on_epoch=True)
        
        rse = self.val_rse(y_hat, y)
        self.log('val_rse', rse, on_epoch=True)
        """
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assuming the batch is a tuple (X, Y) and we need only X for prediction
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        
        if self.learning_rate is None:
            optimizer = torch.optim.Adam(self.parameters())
        
            # Define the lambda function for Noam Scheduler (recomended for Transformer models in the paper Attention is All You Need)
            lr_lambda = lambda step: (0.0 if step == 0 else (self.model_size_for_noam_scheduler_formula ** (-0.5)) * min(step ** (-0.5), step * (self.scheduler_warmup_steps ** (-1.5))))
            
            # Define the scheduler
            scheduler_config = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step'
            }
            
            return [optimizer], [scheduler_config]        
        else:
        
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            
            return optimizer
        