import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.enc_in  # Number of input features
        self.output_dim = configs.output_dim  # Number of target variables
        self.hidden_dim = configs.d_model  # Hidden dimension

        # Causal 1D Convolution Layers
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, 
                               out_channels=self.hidden_dim, 
                               kernel_size=3, 
                               padding=2, 
                               dilation=1)
        
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, 
                               out_channels=self.hidden_dim, 
                               kernel_size=3, 
                               padding=4, 
                               dilation=2)
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_dim, 
                               out_channels=self.hidden_dim, 
                               kernel_size=3, 
                               padding=8, 
                               dilation=4)

        # Compute the output sequence length dynamically
        dummy_input = torch.zeros(1, self.input_dim, self.seq_len)
        with torch.no_grad():
            dummy_out = self.conv1(dummy_input)
            dummy_out = self.conv2(dummy_out)
            dummy_out = self.conv3(dummy_out)
            self.final_seq_len = dummy_out.shape[-1]  # Dynamically get output sequence length

        # Fully connected layer: Adjust for multiple target variables
        self.fc = nn.Linear(self.hidden_dim * self.final_seq_len, self.pred_len * self.output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = x.reshape(x.shape[0], -1)  # Flatten dynamically based on computed sequence length
        x = self.fc(x)
        
        return x.view(x.shape[0], self.pred_len, self.output_dim)  # Reshape to (batch, pred_len, num_targets)
    
    def forecast(self, x_enc):
        return self.forward(x_enc)
