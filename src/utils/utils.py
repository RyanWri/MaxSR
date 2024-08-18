import numpy as np
import yaml
import torch
import datetime
import os
import logging.config
from torchvision import transforms
from PIL import Image


# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_torch_model(model, run_id: str, epoch: int) -> None:
    base_dir = "/home/linuxu/Documents/models/MaxSR"
    # Create the run_id folder if it doesn't exist
    os.makedirs(f"{base_dir}/{run_id}", exist_ok=True)
    # save model in our experiment run id with epoch number as filename
    model_path = f"{base_dir}/{run_id}/epoch_{epoch}.pth"
    # save model
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


def setup_logging(logging_config_path: str) -> None:
    with open(logging_config_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)


def save_checkpoint(state, run_id, epoch, keep_last):
    """
    store model state dict as checkpoint, no more than {keep_last} models in folder for memory
    """
    base_dir = "/home/linuxu/Documents/models/MaxSR-Tiny/"
    checkpoint_dir = f"{base_dir}/{run_id}/model-checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the current model state with epoch number
    torch.save(state, f"{checkpoint_dir}/model-epoch-{epoch}.pth")

    # Get all model files sorted by modification time
    checkpoints = sorted(
        os.listdir(checkpoint_dir),
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
    )

    # Keep only the last `keep_last` models
    while len(checkpoints) > keep_last:
        os.remove(os.path.join(checkpoint_dir, checkpoints.pop(0)))


# Load an image and convert it to a tensor
def load_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor
