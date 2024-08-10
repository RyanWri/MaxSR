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


if __name__ == "__main__":
    # generate unique run id for experimentation
    run_id = generate_run_id()
    print(f"Starting training for Run ID: {run_id}")

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
    # Set learning rate to 2 * 10^-4
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function
    criterion = nn.L1Loss()

    # collect losses from batches
    final_losses = []

    # Process each batch
    for batch_index, (lr_patches, hr_patches) in enumerate(data_loader):
        batch_loss, batch_time = process_batch(
            lr_patches=lr_patches,
            hr_patches=hr_patches,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(f"batch {batch_index} time: {batch_time}")
        final_losses.append(batch_loss)

    maxsr_final_mae_loss = calculate_np_mae_loss(final_losses)
    print(f" MaxSR MAE L1Loss is, {maxsr_final_mae_loss}")

    save_torch_model(model, run_id=run_id)
