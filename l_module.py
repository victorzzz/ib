import torch
import torch.nn as nn
import lightning as L
     
class TimeSeriesModule(L.LightningModule):
    def __init__(self, 
                 model:nn.Module,
                 learning_rate:float | None,
                 scheduler_warmup_steps:int,
                 model_size_for_noam_scheduler_formula:int):
        
        super(TimeSeriesModule, self).__init__()

        self.learning_rate = learning_rate
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.model_size_for_noam_scheduler_formula = model_size_for_noam_scheduler_formula
        
        self.model = model

        self.save_hyperparameters(ignore=["model"])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) 
        
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1) 

        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
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
        