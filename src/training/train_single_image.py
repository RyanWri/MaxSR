from utils.utils import load_config, save_torch_model
from patches_extractor.patches_extractor import PairedPatchesDataset
from model.maxsr import MaxSRModel
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import logging
import torch.optim as optim
import time


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]

    # Example usage
    hr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad"
    lr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4"
    dataset = PairedPatchesDataset(hr_dir, lr_dir, hr_patch_size=256, lr_patch_size=64)
    data_loader = DataLoader(dataset, batch_size=1)

    # Assume the model is already defined and loaded
    model = MaxSRModel(config)
    model.train()  # Set the model to training mode

    # Assuming 'model' is your neural network model
    optimizer = optim.Adam(
        model.parameters(), lr=2e-4
    )  # Set learning rate to 2 * 10^-4

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function
    criterion = nn.L1Loss()

    # Prepare to collect losses
    losses = []
    final_losses = []

    logger.info("---------- START TRAINING---------")

    # Process each batch
    for batch_index, (lr_patches, hr_patches) in enumerate(data_loader):
        if batch_index >= 32:
            break

        logger.info("low resoultion patches shape", lr_patches.shape)

        start_time = time.time()
        # Assuming `lr_patch` is your tensor with shape (1, 64, 3, 64, 64)
        num_of_patches = lr_patches.shape[1]
        # Access the second dimension, which has 64 elements
        # Loop through each patch
        for patch_index in range(num_of_patches):
            single_patch = lr_patches[0, patch_index].unsqueeze(
                0
            )  # (3, 64, 64) -> (1, 3, 64, 64)
            hr_single_patch = hr_patches[0, patch_index].unsqueeze(
                0
            )  # (3,256,256) -> (1, 3, 256, 256)
            # Now `single_patch` is ready to be input to your model
            single_patch = single_patch.to(
                device
            )  # Ensure the patch is on the correct device
            hr_single_patch = hr_single_patch.to(device)

            optimizer.zero_grad()  # Clear existing gradients
            output = model(single_patch)  # Process each patch through your model
            print(f"Output shape for patch {patch_index+1}: {output.shape}")

            # If you have a corresponding HR patch for calculating loss, you can do it here
            loss = criterion(output, hr_single_patch)
            print(f"Loss for patch {patch_index+1}: {loss.item()}")
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        batch_loss = np.sum(np.array(losses)) / len(losses)
        final_losses.append(batch_loss)
        # clear image losses for next image
        losses.clear()
        print(f"Loss for batch index {batch_index+1}: {batch_loss}")
        batch_time = time.time() - start_time
        logger.info(f"batch index {batch_index+1} time took: {batch_time}")

    maxsr_final_mae_loss = np.sum(np.array(final_losses)) / len(final_losses)
    print(f" MaxSR MAE L1Loss is, {maxsr_final_mae_loss}")

    save_torch_model(model, version="version-0-0-3")
