import torch.nn as nn
from components.block_attention import BlockAttention
from components.grid_attention import GridAttention
from components.mb_conv_with_se import MBConvSE


class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, in_channels, dim, num_heads=4, block_size=8):
        super(AdaptiveMaxViTBlock, self).__init__()
        self.mbconv_se = MBConvSE(in_channels, dim)
        self.block_attention = BlockAttention(dim, num_heads, block_size)
        self.grid_attention = GridAttention(dim, num_heads)

    def forward(self, x):
        x = self.mbconv_se(x)
        x = self.block_attention(x)
        x = self.grid_attention(x)
        return x
