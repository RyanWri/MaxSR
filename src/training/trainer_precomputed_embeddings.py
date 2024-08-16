from torch.cuda.amp import autocast, GradScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from model.max_sr_model import MaxSRModel
from utils.utils import (
    calculate_np_mae_loss,
    generate_run_id,
    load_config,
    save_torch_model,
)
from patches_extractor.embedding import PatchEmbedding
import time
from preprossecing.lr_hr_dataset import PrecomputedEmbeddingDataset
from training.cuda_cleaner import clean_cuda_memory_by_threshold


if __name__ == "__main__":
    # log time
    start = time.time()

    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]

    # High resoultion folder (3,2048,2048)
    # Low resolution folder (3,512,512)
    hr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad"
    lr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4"

    # Initialize the PatchEmbedding module
    patch_embedding_model = PatchEmbedding(
        patch_size=config["patch_size"],
        emb_size=config["emb_size"],
        num_patches=config["num_patches"],
    )

    # Move to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_embedding = patch_embedding_model.to(device)

    # Get pairs of LR images and HR images
    dataset = PrecomputedEmbeddingDataset(lr_dir, hr_dir, patch_embedding_model)
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    # Instantiate model
    model = MaxSRModel(config).to(device)

    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # generate unique run id for experimentation
    run_id = generate_run_id()
    print(f"Starting training for Run ID: {run_id}")

    model.train()

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()
    epochs = 10

    for epoch in range(1, epochs + 1):
        # lr_image is low resolution, hr_image is high resolution
        for index, (lr_image_embedded, hr_image) in enumerate(data_loader):
            print(f"processing image {index+1} in epoch {epoch}")
            image_start_time = time.time()

            lr_image_embedded = lr_image_embedded.to(device)
            hr_image = hr_image.to(device)
            optimizer.zero_grad()

            # Enable mixed precision with autocast
            with autocast():
                # Run through the model
                output = model(lr_image_embedded)
                # Compute MAE loss
                loss = criterion(output, hr_image)

            # Backward pass and optimization with mixed precision
            scaler.scale(loss).backward()  # Scales the loss and backpropagates
            scaler.step(optimizer)  # Unscales gradients and updates parameters
            scaler.update()  # Updates the scaler for next iteration

            batch_time = time.time() - image_start_time
            print(f"image {index + 1}/800 took {batch_time:.2f} seconds")

            if clean_cuda_memory_by_threshold(memory_threshold_gb=6.0):
                print("Clearing GPU cache")
                torch.cuda.empty_cache()

        # save model after each epoch
        save_torch_model(model, run_id=run_id, epoch=epoch)
        print(f"Epoch [{epoch}/10], Loss: {loss.item():.4f}")

    print("Training complete.")
