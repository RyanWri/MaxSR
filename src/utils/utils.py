import numpy as np
import yaml
import torch
import datetime
import os


# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_torch_model(model, run_id: str) -> None:
    version = "version-0-0-1"
    base_dir = "/home/linuxu/Documents/models/MaxSR"
    # Create the run_id folder if it doesn't exist
    os.makedirs(f"{base_dir}/{run_id}", exist_ok=True)
    # save model in our experiment run id
    model_path = f"{base_dir}/{run_id}/{version}.pth"
    torch.save(model.state_dict(), model_path)


def generate_run_id():
    now = datetime.datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    return run_id


def calculate_np_mae_loss(losses: list) -> float:
    try:
        return np.sum(np.array(losses)) / len(losses)
    except ValueError as err:
        print(f"error trying to calculate MAE on an empty array")
