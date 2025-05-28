import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class PETformer(nn.Module):
    def __init__(self, configs):
        super(PETformer, self).__init__()

        # Required parameters (from configs)
        self.seq_len = getattr(configs, "seq_len", 96)
        self.pred_len = getattr(configs, "pred_len", 48)
        self.enc_in = getattr(configs, "enc_in", 1)
        self.d_model = getattr(configs, "d_model", 64)
        self.dropout_rate = getattr(configs, "dropout", 0.1)
        self.task_name = getattr(configs, "task_name", "short_term_forecast")
        
        # Add out_dim parameter for multiple target variables
        self.out_dim = getattr(configs, "output_dim", self.enc_in)  # Default to enc_in if not specified

        # PETformer-specific parameters (default values added)
        self.ff_factor = getattr(configs, "ff_factor", 2)
        self.n_heads = getattr(configs, "n_heads", 8)
        self.e_layers = getattr(configs, "e_layers", 3)
        self.win_len = getattr(configs, "win_len", 12)
        self.win_stride = getattr(configs, "win_stride", 6)
        self.channel_attn = getattr(configs, "channel_attn", 0)
        self.attn_type = getattr(configs, "attn_type", 0)
        
        # Compute token sizes
        self.input_token_size = (self.seq_len - self.win_len) // self.win_stride + 1
        self.output_token_size = self.pred_len // self.win_len
        self.all_token_size = self.input_token_size + self.output_token_size
        
        # Model layers
        self.revin = RevIN(self.enc_in, affine=False, subtract_last=True)
        self.mapping = nn.Linear(self.win_len, self.d_model)
        self.placeholder = nn.Parameter(torch.randn(self.d_model))
        self.positionEmbedding = nn.Embedding(self.all_token_size, self.d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.n_heads, batch_first=True,
                dim_feedforward=self.d_model * self.ff_factor,
                dropout=self.dropout_rate
            ) for _ in range(self.e_layers)
        ])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.all_token_size) for _ in range(self.e_layers)])
        
        # Output layers
        self.predict = nn.Linear(self.d_model, self.win_len)
        
        # Add a final projection layer to map from enc_in to out_dim
        if self.out_dim != self.enc_in:
            self.output_proj = nn.Linear(self.enc_in, self.out_dim)
        
        # Attention mask
        self.mask = self._create_attention_mask()
        
        # Channel attention
        if self.channel_attn:
            self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1, batch_first=True, dropout=0.5)
            self.channel_norm = nn.BatchNorm1d(self.enc_in)
        if self.channel_attn in [2, 3]:
            self.channel_pe = nn.Embedding(self.enc_in, self.d_model)
    
    def _create_attention_mask(self):
        mask = torch.full((self.all_token_size, self.all_token_size), True)
        if self.attn_type == 0:
            mask.fill_(False)
        elif self.attn_type == 1:
            mask[:, :self.input_token_size] = False
            mask[torch.eye(self.all_token_size, dtype=bool)] = False
        elif self.attn_type == 2:
            mask[self.input_token_size:, :] = False
            mask[torch.eye(self.all_token_size, dtype=bool)] = False
        elif self.attn_type == 3:
            mask[self.input_token_size:, :self.input_token_size] = False
            mask[torch.eye(self.all_token_size, dtype=bool)] = False
        return mask
    
    def forward(self, x_enc):
        # x_enc: [B, L, D]
        # Normalize input features
        x_norm = self.revin(x_enc, 'norm')  # [B, L, D]
        x = x_norm.permute(0, 2, 1)  # [B, D, L]
        
        # Create tokens from input sequence
        input_tokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride))
        all_tokens = torch.cat(
            (input_tokens, self.placeholder.repeat(x_enc.size(0), self.enc_in, self.output_token_size, 1)), dim=2
        )
        
        # Add positional embeddings
        batch_index = torch.arange(self.all_token_size).expand(x_enc.size(0) * self.enc_in, self.all_token_size).to(x.device)
        all_tokens = all_tokens.view(-1, self.all_token_size, self.d_model) + self.positionEmbedding(batch_index)
        
        # Pass through transformer layers
        for i in range(self.e_layers):
            all_tokens = self.batch_norms[i](self.transformer_layers[i](all_tokens, src_mask=self.mask.to(x.device))) + all_tokens
        
        # Get output tokens
        output_tokens = all_tokens.view(-1, self.enc_in, self.all_token_size, self.d_model)[:, :, self.input_token_size:, :]
        
        # Apply attention if specified
        if self.channel_attn == 1:
            output_tokens = self._apply_self_attention(output_tokens)
        elif self.channel_attn in [2, 3]:
            output_tokens = self._apply_cross_attention(output_tokens, x_enc)
        
        # Generate predictions for input features
        enc_out = self.predict(output_tokens).view(-1, self.enc_in, self.pred_len).permute(0, 2, 1)  # [B, pred_len, enc_in]
        
        # Denormalize back to original scale for input features
        dec_out = self.revin(enc_out, 'denorm')
        
        # Project to output dimensions if necessary
        if self.out_dim != self.enc_in:
            return self.output_proj(dec_out)  # [B, pred_len, out_dim]
        else:
            return dec_out  # [B, pred_len, enc_in]
    
    def _apply_self_attention(self, output_tokens):
        output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
        output_tokens = self.channel_norm(self.attn(output_tokens, output_tokens, output_tokens)[0]) + output_tokens
        return output_tokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
    
    def _apply_cross_attention(self, output_tokens, x_enc):
        channel_index = torch.arange(self.enc_in).expand(x_enc.size(0) * self.output_token_size, self.enc_in).to(x_enc.device)
        query = self.channel_pe(channel_index) if self.channel_attn == 2 else output_tokens
        output_tokens = output_tokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
        output_tokens = self.channel_norm(self.attn(query, query, output_tokens)[0]) + output_tokens
        return output_tokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
    
    def forecast(self, x_enc):
        return self.forward(x_enc)
    
    def imputation(self, x_enc):
        return self.forward(x_enc)
    
    def anomaly_detection(self, x_enc):
        return self.forward(x_enc)
    
    def classification(self, x_enc):
        enc_out = self.forward(x_enc)
        return enc_out.reshape(enc_out.shape[0], -1)
    
    def execute(self, x_enc):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
        return None