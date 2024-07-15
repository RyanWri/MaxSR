import torch
import torch.nn as nn


class ReconstructionBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride_size=1,
        expand_rate=4,
        upscale_factor=2,
    ):
        super(ReconstructionBlock, self).__init__()
        hidden_dim = int(expand_rate * in_dim)

        # Pointwise convolution to expand channels
        self.pw_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False)

        # 3x3 Depthwise Conv:
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Pixel Shuffle for upscaling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Final convolution layer
        self.conv = nn.Conv2d(
            hidden_dim // (upscale_factor**2),
            out_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.pw_conv(x)  # Expand channels
        x = self.dw_conv(x)  # Apply depthwise convolution
        x = self.pixel_shuffle(x)  # Upscale the input tensor
        x = self.conv(x)  # Refine the upscaled features
        return x


# Example usage:
# Assuming in_dim is 64, out_dim is 3 (for RGB), and upscale_factor is 2
reconstruction_block = ReconstructionBlock(in_dim=64, out_dim=3, upscale_factor=2)
input_tensor = torch.rand((1, 64, 32, 32))  # Example input tensor
output_tensor = reconstruction_block(input_tensor)
print(output_tensor.shape)  # Should be (1, 3, 64, 64) if upscale_factor is 2
