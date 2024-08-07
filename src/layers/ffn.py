import torch
import torch.nn as nn


class FFN(nn.Module):
    """
    FFN is FeedForwardNetwork
    """

    def __init__(self, config: dict):
        super(FFN, self).__init__()
        in_channels, hidden_dim = config["ffn_in_channels"], config["ffn_out_channels"]
        hidden_dim = config["hidden_dim"]
        dropout = 0.1
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x
