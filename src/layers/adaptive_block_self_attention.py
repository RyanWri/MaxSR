import torch
import torch.nn as nn


class AdaptiveBlockSelfAttention(nn.Module):
    def __init__(self, config):
        super(AdaptiveBlockSelfAttention, self).__init__()
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.out_channels // self.num_heads
        assert (
            self.out_channels % self.num_heads == 0
        ), "out_channels must be divisible by num_heads"

        self.query = nn.Linear(self.in_channels, self.out_channels)
        self.key = nn.Linear(self.in_channels, self.out_channels)
        self.value = nn.Linear(self.in_channels, self.out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape input to [batch, height * width, channels]
        x = x.view(B, C, H * W).transpose(1, 2)  # Now shape is [B, H*W, C]

        q = (
            self.query(x)
            .view(B, H * W, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.key(x)
            .view(B, H * W, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.value(x)
            .view(B, H * W, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, H * W, C)

        # Reshape output back to [batch, channels, height, width]
        out = out.transpose(1, 2).view(B, C, H, W)

        return out
