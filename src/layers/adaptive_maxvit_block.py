from mb_conv_with_se import MBConv
from adaptive_block_sa import AdaptiveBlockSelfAttention
from adaptive_grid_attention import AdaptiveGridSelfAttention
import torch.nn as nn


class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, in_channels, config):
        super(AdaptiveMaxViTBlock, self).__init__()
        self.mb_conv = MBConv(in_channels, config)
        self.adaptive_block_sa = AdaptiveBlockSelfAttention(in_channels, config)
        self.adaptive_grid_sa = AdaptiveGridSelfAttention(in_channels, config)

    def forward(self, x):
        x = self.mb_conv(x)
        x = self.adaptive_block_sa(x)
        x = self.adaptive_grid_sa(x)
        return x
