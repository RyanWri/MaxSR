import os
import torch
import torch.nn as nn
from model.maxsr import MaxSRModel
from utils.utils import load_config, save_torch_model, generate_run_id, setup_logging
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

if __name__ == "__main__":
    # Load configuration
    config = load_config()["model_config"]

    # Call this at the start of your application to turn on/off logs
    setup_logging(os.path.join(os.getcwd(), "config", "logging_conf.yaml"))

    # Define our MaxSR model using config provided
    model = MaxSRModel(config)

    # define HR dir and LR_bicubic dir
    hr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad"
    lr_dir = "/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4"

    # Create matching pairs (x,y) x=LR_image y=HR_image
    dataset = PairedPatchesDataset(hr_dir, lr_dir, hr_patch_size=256, lr_patch_size=64)
    # Each batch get single image splitted to 64 patches each
    data_loader = DataLoader(dataset, batch_size=1)

    # generate unique run id for experimentation
    run_id = generate_run_id()
    print(f"Starting training for Run ID: {run_id}")

    # Set the model to training mode
    model.train()

    # # Set learning rate to 2 * 10^-4
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # Define the loss function, L1Loss is MAE
    criterion = nn.L1Loss()

    # set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to appropriate device
    model.to(device)
