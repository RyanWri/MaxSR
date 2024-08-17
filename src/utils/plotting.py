import torch
import matplotlib.pyplot as plt
from PIL import Image


def show_patches(hr_patches, lr_patches, patch_index):
    """
    Displays HR and LR patches side-by-side
    You must specify patch index to display
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    hr_patch = hr_patches[0, patch_index].permute(
        1, 2, 0
    )  # Rearrange dimensions for plotting
    axs[0].imshow(hr_patch)
    axs[0].axis("off")  # Hide axes
    axs[0].set_title("HR Patch")

    # LR patches
    lr_patch = lr_patches[0, patch_index].permute(1, 2, 0)
    axs[1].imshow(lr_patch)
    axs[1].axis("off")
    axs[1].set_title("LR Patch")

    plt.show(block=True)


def save_tensor_as_image(tensor, file_path):
    """
    Takes a tensor with shape (1, 3, 2048, 2048), converts it to an RGB image,
    plots the image, and saves it to a specified file path.
    MaxSR model output is (1,3,2048,2048), do not forget to detach it from GPU

    Args:
    tensor (torch.Tensor): Input tensor with shape (1, 3, 2048, 2048).
    file_path (str): Path where the image will be saved.
    """
    # Check if the input tensor has the correct shape
    if tensor.shape != (1, 3, 2048, 2048):
        raise ValueError("Input tensor must have shape (1, 3, 2048, 2048)")

    # Squeeze the tensor to remove the batch dimension
    image_tensor = tensor.squeeze(0)  # This changes the shape to (3, 2048, 2048)

    # Convert the tensor to a PIL Image
    # Transpose the tensor to have (H, W, C) format from (C, H, W)
    image_tensor = image_tensor.permute(1, 2, 0)
    # Scale the tensor from 0-1 (if necessary) and convert to uint8
    image_tensor = image_tensor * 255  # Assuming the tensor is scaled between 0 and 1
    image_tensor = image_tensor.detach().cpu().byte().numpy()
    image = Image.fromarray(image_tensor)

    # Plotting the image using matplotlib
    plt.imshow(image)
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.title("MaxSR Reconstructed Image")

    # Save the image
    image.save(file_path)
