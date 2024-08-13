import torch
import torch.nn as nn
from components.amtb.mbconv_with_se import MBConvWithSE
from components.amtb.adaptive_block_attention import AdaptiveBlockAttention
from components.amtb.adaptive_grid_attention import AdaptiveGridAttention

class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads=8):
        super(AdaptiveMaxViTBlock, self).__init__()
        # Initialize the three components
        self.mb_conv_se = MBConvWithSE(in_channels=in_features, out_channels=out_features)
        self.adaptive_block_attention = AdaptiveBlockAttention(in_features, hidden_features, heads=heads)
        self.adaptive_grid_attention = AdaptiveGridAttention(in_features, hidden_features, heads=heads)

    def forward(self, x):
        # 1. MBConv with Squeeze-and-Excitation
        x = self.mb_conv_se(x, x)  # Input and F0 are the same for the residual connection

        # 2. Adaptive Block Attention
        x = self.adaptive_block_attention(x)

        # 3. Adaptive Grid Attention
        x = self.adaptive_grid_attention(x)

        return x

# Example usage
in_features = 128  # Number of features per patch
hidden_features = 512  # Hidden layer size in FFN for attention blocks
out_features = 128  # Output features after MBConvWithSE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adaptive_maxvit_block = AdaptiveMaxViTBlock(in_features, hidden_features, out_features)
adaptive_maxvit_block = adaptive_maxvit_block.to(device)  # Ensure it's on the correct device

# Assuming 'output' from the previous component
output = torch.randn(128, 64, 128).to(device)  # Simulating input
final_output = adaptive_maxvit_block(output)
print("Final Adaptive MaxViT Block output shape:", final_output.shape)
