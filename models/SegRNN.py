import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Implements a moving average filter, commonly used in time series forecasting to smooth data.
    kernel_size defines how many past values contribute to the smoothing.
    Uses 1D average pooling to compute the moving average.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    Modified to support multiple target variables
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.out_dim = configs.output_dim  

        self.seg_len = min(configs.seg_len, configs.pred_len)

        if self.seq_len % self.seg_len != 0:
            self.seg_len = self.seq_len // (self.seq_len // self.seg_len) 

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = max(1, self.pred_len // self.seg_len)

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        self.output_projection = nn.Linear(self.enc_in, self.out_dim)

    def encoder(self, x):
        batch_size = x.size(0)

        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) 

        if x.shape[2] % self.seg_len != 0:
            raise ValueError(f"seg_len={self.seg_len} does not divide input sequence length {x.shape[2]}!")

        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        _, hn = self.rnn(x) 

        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) 

        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        y = y.permute(0, 2, 1) + seq_last 

        return y

    def forecast(self, x_enc):
        dec_out = self.encoder(x_enc)  
        
        dec_out = self.output_projection(dec_out) 
        
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]