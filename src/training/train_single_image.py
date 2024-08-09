from utils.utils import load_config
from patches_extractor.patches_extractor import PairedPatchesDataset
from model.maxsr import MaxSRModel
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import logging


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
    dataset = PairedPatchesDataset(hr_dir, lr_dir)
    data_loader = DataLoader(dataset, batch_size=1)

    # Assume the model is already defined and loaded
    model = MaxSRModel(config)
    model.eval()  # Set the model to evaluation mode if not training

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function
    criterion = nn.L1Loss()

    # Prepare to collect losses
    losses = []

    # Process each batch
    for i, (lr_patches, hr_patches) in enumerate(data_loader):
        logger.info("low resoultion patches shape", lr_patches.shape)
        criterion = nn.L1Loss()

        # Assuming `lr_patch` is your tensor with shape (1, 64, 3, 64, 64)
        num_of_patches = lr_patches.shape[1] 
        # Access the second dimension, which has 64 elements
        # Loop through each patch
        for i in range(num_of_patches):  
            single_patch = lr_patches[0, i].unsqueeze(0)  # (3, 64, 64) -> (1, 3, 64, 64)
            hr_single_patch = hr_patches[0,i].unsqueeze(0) # (3,256,256) -> (1, 3, 256, 256)
            # Now `single_patch` is ready to be input to your model
            single_patch = single_patch.to(device)  # Ensure the patch is on the correct device
            hr_single_patch = hr_single_patch.to(device)
            

            with torch.no_grad():  # Assuming you're in evaluation mode
                output = model(single_patch)  # Process each patch through your model
                print(f'Output shape for patch {i+1}: {output.shape}')

                # If you have a corresponding HR patch for calculating loss, you can do it here
                # Example: Assuming `hr_patches` is your high-res patches tensor
                hr_patch = hr_patches[0, i].unsqueeze(0).to(device)
                loss = criterion(output, hr_single_patch)  # Assuming `criterion` is defined
                print(f'Loss for patch {i+1}: {loss.item()}')