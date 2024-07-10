import torch
import torch.nn as nn
from layers.ffn import FFN


class AdaptiveBlockSelfAttention(nn.Module):
    def __init__(self, in_channels, block_size, config):
        super(AdaptiveBlockSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.ffn = FFN(in_channels, config.hidden_dim)

    def forward_adaptive_block_sa(self, x):
        B, C, H, W = x.shape
        assert (
            H % self.block_size == 0 and W % self.block_size == 0
        ), "Height and Width must be divisible by block size"

        # Unfold the input into blocks
        blocks = x.unfold(2, self.block_size, self.block_size).unfold(
            3, self.block_size, self.block_size
        )
        B, C, num_blocks_h, num_blocks_w, block_h, block_w = blocks.shape
        blocks = blocks.contiguous().view(B, C, -1, block_h * block_w)

        # Compute Q, K, V matrices
        Q = self.query_conv(blocks).view(B, C, -1, block_h * block_w)
        K = self.key_conv(blocks).view(B, C, -1, block_h * block_w)
        V = self.value_conv(blocks).view(B, C, -1, block_h * block_w)

        # Compute attention scores
        attention_scores = torch.matmul(Q.transpose(-2, -1), K) / (C**0.5)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, V)
        out = out.view(B, C, num_blocks_h, num_blocks_w, block_h, block_w)
        out = out.contiguous().view(B, C, H, W)

        return out

    def forward(self, x):
        # x = x + output(Adaptive block-sa(x))
        x = x + self.forward_adaptive_block_sa(x)
        # x = x + output(FFN(x))
        x = x + self.ffn(x)
        return x
