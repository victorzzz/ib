import torch.nn as nn

class FFModel(nn.Module):
    
    def __init__(self, 
                 input_dim:int, 
                 seq_len:int,
                 out_dim:int, 
                 num_layers:int,
                 denominators:tuple[int, int, int, int], 
                 dropout:float,
                 ff_use_batchnorm:bool):
        
        super(FFModel, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.denominators = denominators
        self.out_dim = out_dim
        self.dropout = dropout
        self.ff_use_batchnorm = ff_use_batchnorm

        layer_input_dim = self.input_dim * self.seq_len
        
        layer_out_dim = layer_input_dim * 4

        self.sequential = nn.Sequential()
        
        self.sequential.append(nn.Linear(layer_input_dim, layer_out_dim))
        if self.ff_use_batchnorm:
            self.sequential.append(nn.BatchNorm1d(layer_out_dim))
        else:
            self.sequential.append(nn.LayerNorm(layer_out_dim))
        self.sequential.append(nn.ReLU())
        self.sequential.append(nn.Dropout(p=self.dropout))
        
        layer_input_dim = layer_out_dim
        
        for i in range(0, len(self.denominators)):
            layer_out_dim = layer_input_dim // self.denominators[i]
            self.sequential.append(nn.Linear(layer_input_dim, layer_out_dim))
            if self.ff_use_batchnorm:
                self.sequential.append(nn.BatchNorm1d(layer_out_dim))
            else:
                self.sequential.append(nn.LayerNorm(layer_out_dim))
            self.sequential.append(nn.ReLU())
            self.sequential.append(nn.Dropout(p=self.dropout))
            layer_input_dim = layer_out_dim
            
        for i in range(0, self.num_layers):
            self.sequential.append(nn.Linear(layer_input_dim, layer_input_dim))
            if self.ff_use_batchnorm:
                self.sequential.append(nn.BatchNorm1d(layer_out_dim))
            else:
                self.sequential.append(nn.LayerNorm(layer_out_dim))
            self.sequential.append(nn.ReLU())
            self.sequential.append(nn.Dropout(p=self.dropout))
            
        self.sequential.append(nn.Linear(layer_input_dim, self.out_dim))

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_variables]
        batch_size, _, _ = x.size()        
        x = x.view(batch_size, -1)
        
        x = self.sequential(x)
        return x
  
    