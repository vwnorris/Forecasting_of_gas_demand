import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANAD(nn.Module):
    def __init__(self, configs):
        super(KANAD, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in
        self.output_dim = configs.output_dim 
        self.hidden_dim = configs.d_model
        self.n_freq = 10

        # Time encoding for Fourier basis
        t = torch.arange(self.seq_len).float() / self.seq_len
        self.register_buffer('t', t)


        fourier_basis = [torch.ones_like(t)]
        for n in range(1, self.n_freq + 1):
            fourier_basis.append(torch.cos(2 * math.pi * n * t))
            fourier_basis.append(torch.sin(2 * math.pi * n * t))
        self.register_buffer('fourier_basis', torch.stack(fourier_basis, dim=1))  


        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.fc_coeff = nn.Linear(self.seq_len, self.fourier_basis.size(1))  
        self.recon_project = nn.Linear(self.fourier_basis.size(1), self.seq_len)
        self.output_project = nn.Linear(self.seq_len, self.pred_len)

        self.output_dim_project = nn.Linear(self.input_dim, self.output_dim)

    def forecast(self, x_enc):
        seq_last = x_enc[:, -1:, :].detach()
        x = x_enc - seq_last
        x = x.permute(0, 2, 1) 

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))

        x = x.mean(dim=1)  

        coeffs = self.fc_coeff(x)  
        recon = self.recon_project(coeffs) 
        pred = self.output_project(recon) 
        pred = pred.unsqueeze(-1)  
        
        pred_expanded = pred.expand(-1, -1, self.input_dim)  
        pred_with_residual = pred_expanded + seq_last 
        pred_final = self.output_dim_project(pred_with_residual)  

        return pred_final

    def forward(self, x_enc):
        return self.forecast(x_enc)