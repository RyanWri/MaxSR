import matplotlib.pyplot as plt


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


def plot_image(tensor):
    """
    Plot a tensor as an image.

    Args:
    tensor (Tensor): Image tensor to plot, shape (channels, height, width).
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()  # Move tensor to CPU if it's on GPU

    # Normalize the tensor to [0, 1] for displaying purposes if it's not already
    # tensor = tensor.float()  # Ensure tensor is float for accurate division
    # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Change from CxHxW to HxWxC for plotting with matplotlib
    tensor = tensor.permute(1, 2, 0)  # Convert to HxWxC

    # Ensure no channel is above 1 (can happen due to numerical precision issues)
    # tensor.clamp_(0, 1)

    plt.imshow(tensor.numpy())  # Convert to numpy array and plot
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.savefig(f"/home/linuxu/Documents/test.png")
