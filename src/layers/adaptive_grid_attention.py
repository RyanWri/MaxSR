import torch.nn as nn


class GridAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(GridAttention, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels),
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)  # Transpose for attention: [B, C, H*W] -> [H*W, B, C]
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = x + x_attn  # Residual connection
        x = self.norm2(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn  # Residual connection
        x = x.permute(1, 2, 0)  # Transpose back: [H*W, B, C] -> [B, C, H*W]
        return x
