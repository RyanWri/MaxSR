import numpy as np
import yaml


# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def extract_patches(img, patch_size=64, stride=64):
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    return patches.contiguous().view(-1, img.size(1), patch_size, patch_size)
