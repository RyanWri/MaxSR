import torch
import torch.nn.functional as F
import torch.nn as nn


class AdaptiveGridAttention(nn.Module):
    def __init__(self, channels, head_count, block_size=8):
        super(AdaptiveGridAttention, self).__init__()
        self.head_count = head_count
        self.block_size = block_size
        self.scale = (channels // head_count) ** -0.5
        self.attention = nn.MultiheadAttention(channels, head_count)

    def forward(self, x):
        b, c, h, w = x.shape
        grid_size = (
            h + self.block_size - 1
        ) // self.block_size  # Ensures coverage of entire dimension

        # Pad if necessary to make the height and width divisible by grid_size
        pad_h = grid_size * self.block_size - h
        pad_w = grid_size * self.block_size - w
        x = F.pad(x, (0, pad_w, 0, pad_h), "constant", 0)

        # Reshape and permute to fit the attention module's input expectations
        x = x.view(b, c, grid_size, self.block_size, grid_size, self.block_size)
        x = x.permute(0, 2, 4, 3, 5, 1)  # B, grid_H, grid_W, block_H, block_W, C
        x = x.reshape(b * grid_size * grid_size, self.block_size * self.block_size, c)

        x = x * self.scale
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.view(
            b, grid_size, grid_size, self.block_size, self.block_size, c
        )
        attn_output = attn_output.permute(0, 5, 1, 3, 2, 4).reshape(
            b, c, grid_size * self.block_size, grid_size * self.block_size
        )

        # Crop back to the original size if padding was added
        if pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :, :h, :w]

        return attn_output
