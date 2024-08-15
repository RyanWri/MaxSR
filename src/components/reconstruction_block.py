import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ReconstructionBlock(nn.Module):
    """Decodes embedded patches into a large image using pixel shuffle, optimized for GPU usage."""

    def __init__(
        self,
        embed_dim=768,
        num_patches=64,
        final_image_size=2048,
        patch_target_size=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.final_image_size = final_image_size
        self.patch_target_size = patch_target_size

        # Calculate the size before pixel shuffle
        scale_factor = self.patch_target_size // 64  # 256 / 64 = 4
        intermediate_channels = 3 * scale_factor**2  # 3 * 4^2 = 48

        # Linear projection to intermediate size
        self.fc = nn.Linear(
            embed_dim, intermediate_channels * 64 * 64
        )  # 48 channels for a 64x64 grid

        # Pixel shuffle to reach 256x256
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = x[0].unsqueeze(0)
        # x shape: (batch_size, num_patches, embed_dim)
        batch_size = x.shape[0]

        # Linear projection to spatial dimensions
        x = self.fc(x)  # (batch_size, num_patches, 48*64*64)
        x = x.view(
            batch_size, self.num_patches, 48, 64, 64
        )  # reshape to (batch_size, num_patches, C, H, W)

        # Apply pixel shuffle per patch
        x = x.view(
            -1, 48, 64, 64
        )  # reshape for pixel shuffle (batch_size*num_patches, C, H, W)
        x = self.pixel_shuffle(x)  # (batch_size*num_patches, 3, 256, 256)

        # Reassemble patches into an image
        x = x.view(batch_size, 8, 8, 3, 256, 256)  # assume an 8x8 grid of patches
        x = x.permute(
            0, 3, 1, 4, 2, 5
        ).contiguous()  # move the 3 (RGB) next to batch size, and prepare for merge
        x = x.view(batch_size, 3, 2048, 2048)  # merge into final image size

        return x
