import numpy as np
import yaml
import torch


# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_torch_model(model, version) -> None:
    base_dir = "/home/linuxu/Documents/models/MaxSR"
    model_path = f"{base_dir}/{version}.pth"
    torch.save(model.state_dict(), model_path)
