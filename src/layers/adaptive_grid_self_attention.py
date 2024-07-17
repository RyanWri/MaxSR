import torch
import torch.nn as nn
from src.layers.ffn import FFN


class AdaptiveGridSelfAttention(nn.Module):
    def __init__(self, in_channels, grid_size, config):
        super(AdaptiveGridSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.ffn = FFN(in_channels, config.hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H % self.grid_size == 0 and W % self.grid_size == 0
        ), "Height and Width must be divisible by grid size"

        grid_h = H // self.grid_size
        grid_w = W // self.grid_size

        def transform_conv(conv_layer):
            transformed = (
                conv_layer(x)
                .view(B, C, grid_h, self.grid_size, grid_w, self.grid_size)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B, grid_h * grid_w, C, self.grid_size * self.grid_size)
            )
            return transformed

        q = transform_conv(self.query_conv)
        k = transform_conv(self.key_conv)
        v = transform_conv(self.value_conv)

        attn = torch.einsum("bgck, bgcl -> bgkl", q, k) / (C**0.5)
        attn = self.softmax(attn)
        out = torch.einsum("bgkl, bgcl -> bgck", attn, v)

        out = out.view(B, grid_h, grid_w, C, self.grid_size, self.grid_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        x = x + out
        x = x + self.ffn(x)
        return x
