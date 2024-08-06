import torch.nn as nn


class GridAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(GridAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        B, C, H, W = x.shape
        # Flatten spatial dimensions into one sequence dimension
        x = x.view(B, C, H * W).permute(
            0, 2, 1
        )  # Change shape to (B, H*W, C) for attention

        # Attention expects (L, N, E) where L is the sequence length, N is the batch size, E is the embedding dimension
        x = x.permute(1, 0, 2)  # Permute to (H*W, B, C)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # Permute back to (B, H*W, C)

        # Reshape back to the original spatial dimensions
        attn_output = attn_output.view(B, C, H, W)
        return attn_output
