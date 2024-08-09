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
