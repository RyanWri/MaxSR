import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttention(nn.Module):
    def __init__(self, dim, num_heads, block_size):
        super(BlockAttention, self).__init__()
        self.block_size = block_size
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        B, C, H, W = x.shape
        # Flattening the blocks into sequences
        x = x.view(
            B,
            C,
            H // self.block_size,
            self.block_size,
            W // self.block_size,
            self.block_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(
            -1, self.block_size * self.block_size, C
        )  # (N, L, E)

        # Attention requires (L, N, E)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)

        # Reshape back to original dimensions
        attn_output = attn_output.view(
            B,
            H // self.block_size,
            W // self.block_size,
            self.block_size,
            self.block_size,
            C,
        )
        attn_output = attn_output.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        return attn_output
