import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANAD(nn.Module):
    def __init__(self, configs):
        super(KANAD, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in        # Number of input features
        self.output_dim = configs.output_dim   # Number of target variables
        self.hidden_dim = configs.d_model
        self.n_freq = 10

        # Time encoding for Fourier basis
        t = torch.arange(self.seq_len).float() / self.seq_len
        self.register_buffer('t', t)

        # Create Fourier basis: [1, cos, sin, cos, sin, ...]
        fourier_basis = [torch.ones_like(t)]
        for n in range(1, self.n_freq + 1):
            fourier_basis.append(torch.cos(2 * math.pi * n * t))
            fourier_basis.append(torch.sin(2 * math.pi * n * t))
        self.register_buffer('fourier_basis', torch.stack(fourier_basis, dim=1))  # [seq_len, 2*n_freq+1]

        # Layers
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.fc_coeff = nn.Linear(self.seq_len, self.fourier_basis.size(1))  # match seq_len to num basis
        self.recon_project = nn.Linear(self.fourier_basis.size(1), self.seq_len)
        self.output_project = nn.Linear(self.seq_len, self.pred_len)

        # Final projection to match output dim
        # FIXED: Changed from nn.Linear(1, self.output_dim) to nn.Linear(self.input_dim, self.output_dim)
        self.output_dim_project = nn.Linear(self.input_dim, self.output_dim)

    def forecast(self, x_enc):
        # x_enc: [B, seq_len, input_dim]
        seq_last = x_enc[:, -1:, :].detach()  # [B, 1, input_dim]

        x = x_enc - seq_last
        x = x.permute(0, 2, 1)  # [B, input_dim, seq_len]

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))

        x = x.mean(dim=1)  # [B, seq_len]

        coeffs = self.fc_coeff(x)  # [B, num_basis]
        recon = self.recon_project(coeffs)  # [B, seq_len]
        pred = self.output_project(recon)  # [B, pred_len]
        pred = pred.unsqueeze(-1)  # [B, pred_len, 1]
        
        # FIXED: Need to transpose seq_last before broadcasting
        pred_expanded = pred.expand(-1, -1, self.input_dim)  # [B, pred_len, input_dim]
        
        # Add the difference to last point of sequence for each feature
        pred_with_residual = pred_expanded + seq_last  # [B, pred_len, input_dim]
        
        # Now project to output dimension
        pred_final = self.output_dim_project(pred_with_residual)  # [B, pred_len, output_dim]

        return pred_final

    def forward(self, x_enc):
        return self.forecast(x_enc)