from torch.cuda.amp import autocast, GradScaler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from model.maxsr_super_tiny import MaxSRSuperTiny
from utils.utils import (
    generate_run_id,
    load_config,
    save_checkpoint,
)
import time
from preprossecing.lr_hr_dataset import LRHRDataset
from training.cuda_cleaner import clean_cuda_memory_by_threshold
from model_evaluation.metrics import (
    EarlyStopping,
    calculate_psnr_ssim_metrics,
    log_metrics_to_json,
)


if __name__ == "__main__":
    # log time
    start = time.time()

    # Detect the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_super_tiny.yaml"))
    model_config = config["model_config"]
    paths = config["paths"]

    # High resoultion folder (3,128,128)
    # Low resolution folder (3,64,64)
    hr_dir = "/home/linuxu/Documents/datasets/Tiny_HR"
    lr_dir = "/home/linuxu/Documents/datasets/Tiny_LR"

    # Get pairs of LR images and HR images
    dataset = LRHRDataset(lr_dir, hr_dir)
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    # Instantiate model
    model = MaxSRSuperTiny(model_config).to(device)

    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # generate unique run id for experimentation
    run_id = generate_run_id()
    print(f"Starting training for Run ID: {run_id}")

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()
    epochs = 100000

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=30, min_delta=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        print(f"processing epoch {epoch} / {epochs}")
        epoch_start_time = time.time()
        running_loss, psnr_score, ssim_score = 0.0, 0.0, 0.0
        # lr_image_embedded is low resolution image after patch + embedding, hr_image is high resolution
        for index, (lr_image_embedded, hr_image) in enumerate(data_loader):
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
            if clean_cuda_memory_by_threshold(memory_threshold_gb=6.6):
                print("Clearing GPU cache")
                torch.cuda.empty_cache()

            running_loss += loss.item()
            psnr, ssim = calculate_psnr_ssim_metrics(output, hr_image, device)
            # Calculate PSNR and SSIM for the batch
            psnr_score += psnr
            ssim_score += ssim

        # collect metrics for epoch
        epoch_loss = running_loss / len(data_loader)
        epoch_psnr = psnr_score / len(data_loader)
        epoch_ssim = ssim_score / len(data_loader)
        epoch_time = time.time() - epoch_start_time

        # Log metrics for this iteration
        log_metrics_to_json(
            paths=paths,
            run_id=run_id,
            epoch=epoch,
            loss=epoch_loss,
            psnr=epoch_psnr,
            ssim=epoch_ssim,
            total_time=epoch_time,
        )

        # Check for early stopping
        early_stopping(epoch_loss, epoch_psnr, epoch_ssim)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        # save the model checkpoint if there is an improvment
        if early_stopping.counter == 0:
            print(f"Saving model checkpoint at epoch {epoch}")
            save_checkpoint(
                paths, model.state_dict(), run_id=run_id, epoch=epoch, keep_last=5
            )
            print(f"Model saved at epoch {epoch}")

    print("Training complete.")
