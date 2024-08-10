import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model.maxsr import MaxSRModel
from utils.utils import (
    calculate_np_mae_loss,
    generate_run_id,
    load_config,
    save_torch_model,
)
from patches_extractor.patches_extractor import PairedPatchesDataset
from training.processor import process_batch
import time


def create_dataloader_for_training(
    hr_dir: str, lr_dir, hr_patch_size: int, lr_patch_size: int, batch_size: int
) -> DataLoader:
    dataset = PairedPatchesDataset(hr_dir, lr_dir, hr_patch_size, lr_patch_size)
    return DataLoader(dataset, batch_size=batch_size)


def setup_model(config: dict):
    """
    create MaxSR model
    return the model, Adam optimizer, MAE loss criterion, Device (GPU or CPU)
    """
    model = MaxSRModel(config)
    # Set learning rate to 2 * 10^-4
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # Define the loss function
    criterion = nn.L1Loss()
    # Set device to GPU if relevant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model, optimizer, criterion, device


def run_experiment(data_loader, batch_size, model, optimizer, criterion, device):
    # generate unique run id for experimentation
    run_id = generate_run_id()
    print(f"Starting training for Run ID: {run_id}")

    start = time.time()

    # collect losses from batches
    final_losses = []

    # Process each batch
    for batch_index, (lr_patches, hr_patches) in enumerate(data_loader):
        if batch_index >= 1:
            break

        # Process Batch
        batch_loss, batch_time = process_batch(
            lr_patches=lr_patches,
            hr_patches=hr_patches,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        print(
            f"batch {batch_index} processed {batch_size} Image in time: {batch_time} seconds"
        )
        final_losses.append(batch_loss)

    maxsr_final_mae_loss = calculate_np_mae_loss(final_losses)
    print(f" MaxSR MAE L1Loss is, {maxsr_final_mae_loss}")

    save_torch_model(model, run_id=run_id)
    end = time.time() - start
    print(f"total time for MaxSR training: {end}")


if __name__ == "__main__":
    hr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad"
    lr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4"
    batch_size = 8
    data_loader = create_dataloader_for_training(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        hr_patch_size=256,
        lr_patch_size=64,
        batch_size=batch_size,
    )

    # load model config
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]

    # Create model and relevant params
    model, optimizer, criterion, device = setup_model(config=config)

    # Set the model to training mode
    model.train()
    model.to(device)

    # run Experiment
    run_experiment(
        data_loader=data_loader,
        batch_size=batch_size,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
