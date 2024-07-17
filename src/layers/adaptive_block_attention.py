import torch
import torch.nn as nn
from layers.adaptive_block_sa import AdaptiveBlockSelfAttention
from layers.ffn import FFN


class AdaptiveBlockAttention(nn.Module):
    def __init__(self, config):
        super(AdaptiveBlockAttention, self).__init__()
        # Initialize the Self-Attention component
        self.adaptive_block_sa = AdaptiveBlockSelfAttention(config["self_attention"])

        # Initialize the Feed-Forward Network component
        self.ffn = FFN(config["ffn"])

        # Optionally, add a normalization layer after FFN if required by the architecture specifics
        self.norm = nn.LayerNorm(config["self_attention"]["in_channels"])

    def forward(self, x):
        # Apply self-attention
        x = self.adaptive_block_sa(x)

        # Apply Feed-Forward Network
        x = self.ffn(x)

        # Apply normalization
        x = self.norm(x)

        return x
