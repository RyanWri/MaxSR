import torch
import torch.nn as nn


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(ReconstructionBlock, self).__init__()
        # Prepare for pixel shuffling
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # Followed by a 3x3 convolution
        self.final_conv = nn.Conv2d(
            in_channels // (scale_factor**2), out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # Ensure input is reshaped correctly for 2D convolution and pixel shuffle
        batch_size, num_patches, channels = x.shape
        height_width = int(
            num_patches**0.5
        )  # Assuming num_patches = 64 -> height_width = 8

        # Reshape for PixelShuffle
        x = x.view(batch_size, channels, height_width, height_width)  # (128, 128, 8, 8)

        # Pixel shuffle to upscale the feature map
        x = self.pixel_shuffle(x)  # (128, 8, 32, 32)

        # Apply final 3x3 convolution
        x = self.final_conv(x)  # (128, 3, 32, 32)

        return x


# Example usage
in_channels = 128  # Number of input channels (from HFFB)
out_channels = 3  # Assuming the output is an RGB image
scale_factor = 4  # Upscaling factor

reconstruction_block = ReconstructionBlock(in_channels, out_channels, scale_factor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reconstruction_block = reconstruction_block.to(
    device
)  # Ensure it's on the correct device

# Simulate the HFFB output
hffb_output = torch.randn(128, 64, 128).to(device)  # No incorrect reshaping here

# Pass through the reconstruction block
reconstructed_image = reconstruction_block(hffb_output)
print("Reconstructed image shape:", reconstructed_image.shape)
