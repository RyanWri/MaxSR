import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim, image_size, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * channels, embedding_dim)

        # Positional encodings are learnable parameters, initialized to some values
        self.positional_encodings = nn.Parameter(
            torch.zeros(1, self.num_patches, embedding_dim)
        )

    def forward(self, x):
        # x is shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Extract patches in order (non-randomly)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            batch_size, channels, -1, self.patch_size, self.patch_size
        )
        patches = patches.permute(
            0, 2, 1, 3, 4
        )  # Shape: (batch_size, num_patches, channels, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size, self.num_patches, -1
        )  # Flatten patches

        # Linear embedding
        embedded_patches = self.projection(
            patches
        )  # Shape: (batch_size, num_patches, embedding_dim)

        # Add positional encodings
        embedded_patches += self.positional_encodings

        return embedded_patches


class PatchReconstruction(nn.Module):
    def __init__(self, embedding_dim, patch_size, image_size, channels=3):
        super(PatchReconstruction, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(embedding_dim, patch_size * patch_size * channels)
        self.grid_size = (
            image_size // patch_size
        )  # 8 for a 64x64 image with 8x8 patches

    def forward(self, x):
        # x is shape: (batch_size, num_patches, embedding_dim)
        batch_size, num_patches, embedding_dim = x.shape

        # Reverse the embedding to get patches back
        patches = self.projection(
            x
        )  # Shape: (batch_size, num_patches, patch_size * patch_size * channels)
        patches = patches.view(
            batch_size, num_patches, -1, self.patch_size, self.patch_size
        )  # (batch_size, num_patches, channels, patch_size, patch_size)
        patches = patches.permute(
            0, 2, 1, 3, 4
        )  # Shape: (batch_size, channels, num_patches, patch_size, patch_size)

        # Reshape and reorder patches into the image
        patches = patches.contiguous().view(
            batch_size,
            -1,
            self.grid_size,
            self.patch_size,
            self.grid_size,
            self.patch_size,
        )
        patches = patches.permute(
            0, 1, 2, 4, 3, 5
        )  # Shape: (batch_size, channels, grid_size, grid_size, patch_size, patch_size)
        image = patches.contiguous().view(
            batch_size,
            -1,
            self.grid_size * self.patch_size,
            self.grid_size * self.patch_size,
        )

        return image


class PatchReconstructionWithUpscale(nn.Module):
    def __init__(
        self, embedding_dim, patch_size, image_size, channels=3, scale_factor=2
    ):
        super(PatchReconstructionWithUpscale, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.scale_factor = scale_factor
        self.upscaled_patch_size = patch_size * scale_factor
        self.projection = nn.Linear(
            embedding_dim, channels * (scale_factor**2) * patch_size * patch_size
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.grid_size = (
            image_size // patch_size
        )  # 8 for a 64x64 image with 8x8 patches

    def forward(self, x):
        # x is shape: (batch_size, num_patches, embedding_dim)
        batch_size, num_patches, embedding_dim = x.shape

        # Reverse the embedding to get patches back
        patches = self.projection(
            x
        )  # Shape: (batch_size, num_patches, channels * scale_factor^2 * patch_size^2)
        patches = patches.view(
            batch_size, num_patches, -1, self.patch_size, self.patch_size
        )  # (batch_size, num_patches, channels * scale_factor^2, patch_size, patch_size)
        patches = patches.permute(
            0, 2, 1, 3, 4
        )  # Shape: (batch_size, channels * scale_factor^2, num_patches, patch_size, patch_size)

        # Pixel shuffle to upscale each patch
        patches = patches.view(
            batch_size,
            -1,
            self.grid_size,
            self.patch_size,
            self.grid_size,
            self.patch_size,
        )
        patches = patches.permute(
            0, 1, 2, 4, 3, 5
        )  # Shape: (batch_size, channels * scale_factor^2, grid_size, patch_size, grid_size, patch_size)
        patches = patches.contiguous().view(
            batch_size,
            -1,
            self.patch_size * self.grid_size,
            self.patch_size * self.grid_size,
        )
        upscaled_image = self.pixel_shuffle(
            patches
        )  # Upscales the patches to (batch_size, channels, upscaled_patch_size, upscaled_patch_size)

        # Apply 3x3 convolution for refinement
        upscaled_image = self.conv(
            upscaled_image
        )  # Shape: (batch_size, channels, upscaled_image_size, upscaled_image_size)

        return upscaled_image
