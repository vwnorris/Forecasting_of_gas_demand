import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, configs):
        super(RNN, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in  
        self.output_dim = configs.output_dim 
        self.hidden_dim = configs.d_model 
        self.num_layers = 2 
        
        self.rnn = nn.RNN(input_size=self.input_dim, 
                          hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, 
                          batch_first=True, 
                          nonlinearity='tanh')
        
        self.fc = nn.Linear(self.hidden_dim, self.pred_len * self.output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  
        out, _ = self.rnn(x, h0)  
        
        out = self.fc(out[:, -1, :])  
        
        return out.view(batch_size, self.pred_len, self.output_dim)  
    
    def forecast(self, x_enc):
        return self.forward(x_enc)
