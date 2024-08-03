# File: src/adaptive_grid_attention.py

import torch
import torch.nn as nn
from layers.ffn import FFN
from layers.adaptive_grid_self_attention import AdaptiveGridSelfAttention


class AdaptiveGridAttention(nn.Module):
    def __init__(self, config):
        super(AdaptiveGridAttention, self).__init__()
        # Grid-specific attention mechanism initialization here
        self.grid_attention = AdaptiveGridSelfAttention(config["grid_attention"])
        self.ffn = FFN(
            config["ffn"]["in_channels"],
            config["ffn"]["hidden_dim"],
            config["ffn"]["dropout"],
        )

    def forward(self, x):
        x = self.grid_attention(x)
        x = self.ffn(x)
        return x
