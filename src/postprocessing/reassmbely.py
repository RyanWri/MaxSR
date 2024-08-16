import torch
import numpy as np
import matplotlib.pyplot as plt


def reassemble_image(patches, grid_size=8):
    """
    Reassemble image from patches.

    Args:
        patches (Tensor): Tensor of shape (batch_size, num_patches, num_channels, patch_height, patch_width).
        grid_size (int): The size of the grid (e.g., 8 for 8x8 grid).

    Returns:
        Tensor: Reassembled image of shape (batch_size, num_channels, height, width).
    """
    batch_size, num_channels, patch_height, patch_width = patches.shape
    patches = patches.view(
        batch_size, grid_size, grid_size, num_channels, patch_height, patch_width
    )

    # Permute and reshape to combine patches into full image
    full_image = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    full_image = full_image.view(
        batch_size, num_channels, grid_size * patch_height, grid_size * patch_width
    )

    return full_image


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patches = torch.randn(64, 3, 256, 256).to(
    device
)  # Simulate the output from Reconstruction Block

# Reassemble the patches into the full image
reconstructed_image = reassemble_image(patches)

# Convert to CPU and plot
reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).to("cpu")
plt.imshow(np.clip(reconstructed_image, 0, 1))
plt.show()
