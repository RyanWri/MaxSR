import torch
import torch.nn as nn


class AdaptiveBlockSelfAttention(nn.Module):
    def __init__(self, config):
        super(AdaptiveBlockSelfAttention, self).__init__()
        self.in_channels = config.get("in_channels")
        self.block_size = config.get("block_size", 16)  # Default block size
        self.num_heads = config.get("num_heads", 4)  # Default number of heads
        self.head_dim = self.in_channels // self.num_heads

        if self.in_channels % self.num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

        self.query_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.key_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.value_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.softmax = nn.Softmax(dim=-1)
        self.out_projection = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.norm = nn.LayerNorm(self.in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape and permute the input tensor to form B, num_heads, head_dim, H, W
        x = x.view(B, self.num_heads, self.head_dim, H, W)

        # Unfold the input into blocks
        unfolded = x.unfold(3, self.block_size, self.block_size).unfold(
            4, self.block_size, self.block_size
        )
        B, nH, d, num_blocks_h, num_blocks_w, block_h, block_w = unfolded.shape

        unfolded = unfolded.contiguous().view(B, nH, d, -1, block_h * block_w)

        # Separate Q, K, V
        Q = self.query_conv(unfolded)
        K = self.key_conv(unfolded)
        V = self.value_conv(unfolded)

        # Scaled Dot-Product Attention
        attention_scores = torch.einsum("bnhdk,bnhdj->bnhkj", Q, K) / (d**0.5)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        weighted_values = torch.einsum("bnhkj,bnhdj->bnhdk", attention_weights, V)

        # Reshape to the original unfolded size
        weighted_values = weighted_values.contiguous().view(
            B, self.in_channels, num_blocks_h, num_blocks_w, block_h, block_w
        )
        weighted_values = weighted_values.contiguous().view(B, self.in_channels, H, W)

        # Project back to the original dimension and apply output projection
        out = self.out_projection(weighted_values)
        out = out + x  # Skip connection

        # Normalize
        out = self.norm(out)

        return out
