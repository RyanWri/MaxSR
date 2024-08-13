import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


# Define image postprocessing function
def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    return image


def visualize_feature_maps(feature_maps, save_path, title="F0 Feature Map"):
    # Assume feature_maps is a tensor of shape (B, C, H, W)
    feature_map = feature_maps[0]  # Take the first in the batch for visualization

    # Normalize the feature map for better visualization
    feature_map -= feature_map.min()
    feature_map /= feature_map.max()

    # Convert to numpy and display
    feature_map = feature_map.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=feature_map.shape[0], figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(feature_map[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

def visualize_mbconv_feature_maps(feature_maps, save_path, title="MBConv Feature Map"):
    # Assuming feature_maps shape is (1, C, H, W)
    feature_map = feature_maps[0].detach().cpu()

    fig, axs = plt.subplots(1, feature_map.shape[0], figsize=(20, 2))
    for i, ax in enumerate(axs):
        ax.imshow(feature_map[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

def visualize_attention_feature_maps(feature_maps, save_path, title):
    # Assuming feature_maps shape is (1, C, H, W)
    feature_map = feature_maps[0].detach().cpu()

    fig, axs = plt.subplots(1, feature_map.shape[0], figsize=(20, 2))
    for i, ax in enumerate(axs):
        ax.imshow(feature_map[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def visualize_hffb_feature_maps(feature_maps, save_path, title=" HFFB Feature Map"):
    feature_map = feature_maps[0].detach().cpu()  # Take the first batch
    fig = plt.figure(figsize=(15, 10))  # Assign the figure to a variable
    for i in range(1, 5):  # Visualizing the first 4 feature maps for simplicity
        ax = plt.subplot(1, 4, i)
        ax.imshow(feature_map[i], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)  # Correctly close the figure by passing 'fig'


# Function to visualize the output image
def visualize_RB_output_image(tensor, save_path=None, title="RB Feature Map"):
    image = tensor[0].detach().cpu()  # Take the first in the batch
    image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize the image
    plt.figure(figsize=(8, 8))  # Create a new figure
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure if save_path is provided
    else:
        plt.show()  # Otherwise, display the image
    plt.close()  # Close the figure


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # Clone the tensor to not do changes on it
    image = image.squeeze(0)  # Remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated
