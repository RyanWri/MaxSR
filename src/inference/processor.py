from torchvision import transforms
from PIL import Image
import torch


def load_and_transform_image(image_path, output_size=(512, 512)):
    transform = transforms.Compose(
        [transforms.Resize(output_size), transforms.ToTensor()]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def split_into_patches(image_tensor, patch_size=64):
    # image_tensor shape is expected to be [1, C, H, W]
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches


def inference_image_to_patches(image_path):
    image_tensor = load_and_transform_image(image_path)
    patches = split_into_patches(image_tensor)
    return patches


def reassemble_patches(patches, num_patches_per_row):
    """
    Reassemble patches into a full image assuming they are in row-major order.

    Args:
    patches (Tensor): tensor of shape (num_patches, channels, patch_height, patch_width)
                      where num_patches = num_patches_per_row * num_patches_per_row
    num_patches_per_row (int): number of patches per row (and column) in the full image

    Returns:
    Tensor: the full image tensor of shape (channels, full_image_height, full_image_width)
    """
    # Assuming the patches are correctly ordered and formatted
    # Reshape the flat list of patches into rows of patches
    rows = [
        torch.cat([patches[i + j] for j in range(num_patches_per_row)], dim=2)
        for i in range(0, len(patches), num_patches_per_row)
    ]

    # Concatenate rows vertically
    full_image = torch.cat(rows, dim=1)
    return full_image
