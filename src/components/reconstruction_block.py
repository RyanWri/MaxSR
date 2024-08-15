import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super(ReconstructionBlock, self).__init__()
        # Pixel shuffle with the correct scale factor
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # Final 3x3 convolution to refine the image
        self.conv = nn.Conv2d(
            in_channels // (scale_factor**2), out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        batch_size, emb_size, num_patches, dim = x.shape  # x.shape = (128, 64, 128)

        # Reshape to prepare for PixelShuffle
        x = (
            x.permute(0, 2, 1).contiguous().view(batch_size, dim * emb_size, 8, 8)
        )  # (128, 16384, 8, 8)

        # Apply PixelShuffle to upscale the image
        x = self.pixel_shuffle(x)  # Expected output: (128, 256, 32, 32)

        # Final convolution to get the output image size
        x = self.conv(x)  # Expected output: (128, 3, 256, 256)

        # Reshape to (1, 3, 2048, 2048) after processing all patches together
        x = x.view(batch_size, -1, 2048, 2048)  # Reshape to the final image size
        return x


# Example usage:
in_channels = 128 * 128  # emb_size * dim for PixelShuffle input
out_channels = 3  # RGB channels
reconstruction_block = ReconstructionBlock(in_channels, out_channels).to("cuda")

# Simulate HFFB output for a single image
hffb_output = torch.randn(128, 128, 64, 128).to("cuda")  # (128, 64, 128, 128)

# Process the tensor to reconstruct the image
reconstructed_image = reconstruction_block(hffb_output)

# Check the final image shape
print(
    "Final high-resolution image shape:", reconstructed_image.shape
)  # Should output (128, 3, 2048, 2048)

# Visualize the image (if needed)
reconstructed_image_cpu = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
