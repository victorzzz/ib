import torch.nn as nn

class FFModel(nn.Module):
    
    def __init__(self, 
                 input_dim:int, 
                 out_dim:int, 
                 num_layers:int,
                 denominators:tuple[int, int, int, int], 
                 dropout:float):
        
        super(FFModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.denominators = denominators
        self.out_dim = out_dim
        self.dropout = dropout

        self.sequential = nn.Sequential()
        
        layer_input_dim = self.input_dim
        for i in range(0, len(self.denominators)):
            layer_out_dim = layer_input_dim // self.denominators[i]
            self.sequential.append(nn.Linear(layer_input_dim, layer_out_dim))
            self.sequential.append(nn.LayerNorm(layer_input_dim))
            self.sequential.append(nn.ReLU())
            self.sequential.append(nn.Dropout(p=self.dropout))
            layer_input_dim = layer_out_dim
            
        for i in range(0, self.num_layers):
            self.sequential.append(nn.Linear(layer_input_dim, layer_input_dim))
            self.sequential.append(nn.LayerNorm(layer_input_dim))
            self.sequential.append(nn.ReLU())
            self.sequential.append(nn.Dropout(p=self.dropout))
            
        self.sequential.append(nn.Linear(layer_input_dim, self.out_dim))

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_variables]
        batch_size, _, _ = x.size()        
        x = x.view(batch_size, -1)
        
        x = self.sequential(x)
        return x
  
    