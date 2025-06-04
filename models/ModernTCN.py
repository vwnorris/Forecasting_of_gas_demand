import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features):
        super(RevIN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            self.mean, self.std = mean, std
            return (x - mean) / (std + 1e-5) * self.gamma + self.beta
        elif mode == 'denorm':
            return (x - self.beta) / (self.gamma + 1e-5) * self.std + self.mean

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=16, stride=8):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)

    def forward(self, x):
        return self.conv(x)

class ModernTCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=7, dilation=1, dropout=0.2, ffn_ratio=8):
        super(ModernTCNBlock, self).__init__()

        kernel_size = max(3, (kernel_size // 2) | 1)  

        self.dwconv_large = nn.Conv1d(
            input_dim, input_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size // 2) * dilation,  
            dilation=dilation, 
            groups=input_dim
        )

        self.dwconv_small = nn.Conv1d(
            input_dim, input_dim, 
            kernel_size=5, 
            padding=2, 
            groups=input_dim
        )

        self.norm1 = nn.GroupNorm(1, input_dim)

        self.convffn1 = nn.Conv1d(input_dim, hidden_dim * ffn_ratio, kernel_size=1)
        self.convffn2 = nn.Conv1d(hidden_dim * ffn_ratio, hidden_dim, kernel_size=1, groups=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) if input_dim != hidden_dim else nn.Identity()
        self.norm2 = nn.GroupNorm(1, hidden_dim)

    def forward(self, x):
        res = self.residual(x)

        x1 = self.dwconv_large(x)
        x2 = self.dwconv_small(x)

        if x1.shape[-1] != x2.shape[-1]:
            min_len = min(x1.shape[-1], x2.shape[-1])
            x1 = x1[..., :min_len]
            x2 = x2[..., :min_len]

        x = x1 + x2
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.convffn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.convffn2(x)
        x = self.dropout(x)

        x = x + res
        x = self.norm2(x)
        return x

class ModernTCN(nn.Module):
    def __init__(self, configs, num_blocks=6, kernel_size=51, dropout=0.1, hidden_dim=128, ffn_ratio=8, patch_size=16, stride=8):
        super(ModernTCN, self).__init__()
        self.pred_len = configs.pred_len
        self.hidden_dim = hidden_dim
        self.input_dim = configs.enc_in
        self.output_dim = configs.output_dim
        self.revin = RevIN(self.input_dim)

        self.embedding = PatchEmbedding(self.input_dim, hidden_dim, patch_size=16, stride=2)

        self.blocks = nn.ModuleList([
            ModernTCNBlock(hidden_dim, hidden_dim, 
                           kernel_size=max(3, kernel_size // (2 ** i)),
                           dilation=2 ** i,
                           dropout=dropout, ffn_ratio=ffn_ratio) 
            for i in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        
        self.output_revins = nn.ModuleList([RevIN(1) for _ in range(self.output_dim)])

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        x = self.revin(x.permute(0, 2, 1), mode='norm')  # (batch, input_dim, seq_len)

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 1) 
        x = self.output_layer(x) 

        outputs = []
        
        for i in range(self.output_dim):
            x_feature = x[:, :, i:i+1]

            x_reshaped = torch.zeros(batch_size, 1, x_feature.shape[1], device=x.device)
            x_reshaped[:, 0, :] = x_feature[:, :, 0]
            
            self.output_revins[i].gamma = nn.Parameter(self.revin.gamma[:, 0:1, :].clone())
            self.output_revins[i].beta = nn.Parameter(self.revin.beta[:, 0:1, :].clone())
            self.output_revins[i].mean = self.revin.mean[:, 0:1, :]
            self.output_revins[i].std = self.revin.std[:, 0:1, :]
            
            x_denorm = self.output_revins[i](x_reshaped, mode='denorm')
            outputs.append(x_denorm.permute(0, 2, 1))  

        if len(outputs) > 1:
            final_output = torch.cat(outputs, dim=2)  
        else:
            final_output = outputs[0]

        return final_output[:, -self.pred_len:, :]

    def forecast(self, x):
        """
        Forecast function that ensures output shape matches target shape
        """
        output = self.forward(x)

        if output.shape[2] != self.output_dim:
            if output.shape[2] < self.output_dim:
                padding = torch.zeros(output.shape[0], output.shape[1], 
                                      self.output_dim - output.shape[2], 
                                      device=output.device)
                output = torch.cat([output, padding], dim=2)
            else:
                output = output[:, :, :self.output_dim]
            
        return output