from layers.mb_conv_with_se import MBConv
from layers.adaptive_block_sa import AdaptiveBlockSelfAttention
from layers.adaptive_grid_attention import AdaptiveGridSelfAttention
import torch.nn as nn


class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, config):
        super(AdaptiveMaxViTBlock, self).__init__()
        in_channels = config.get("in_channels", None)
        out_channels = config.get("out_channels", None)
        self.mb_conv = MBConv(in_channels, out_channels)
        self.adaptive_block_sa = AdaptiveBlockSelfAttention(in_channels, config)
        self.adaptive_grid_sa = AdaptiveGridSelfAttention(in_channels, config)

    def forward(self, x):
        x = self.mb_conv(x)
        x = self.adaptive_block_sa(x)
        x = self.adaptive_grid_sa(x)
        return x
