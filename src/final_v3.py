import yaml
from utils.utils import load_config
from patches_extractor.patches_extractor import PairedPatchesDataset
from torch.utils.data import DataLoader
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block.adaptive_maxvit_block import AdaptiveMaxViTBlock
import torch.nn as nn

if __name__ == "__main__":
    # Load configuration
    config = load_config("C:\Afeka\MaxSR\src\config\maxsr_tiny.yaml")["model_config"]

    # Example usage
    hr_dir = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    lr_dir = "C:\datasets\DIV2K\Dataset\DIV2K_train_LR_PAD_BICUBIC_x4"
    dataset = PairedPatchesDataset(hr_dir, lr_dir)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Test loading some data
    for hr_patches, lr_patches in data_loader:
        first_lr_patch = lr_patches[0][32]
        first_hr_patch = hr_patches[0][32]
        break

    sfeb = ShallowFeatureExtractionBlock(config)
    F_minus_1, F0 = sfeb(first_lr_patch)

    stages = nn.ModuleList(
        [
            nn.Sequential(
                AdaptiveMaxViTBlock(config),
                AdaptiveMaxViTBlock(config),
            )
            for _ in range(2)  # Example: 2 stages, each with 2 blocks
        ]
    )

    print(F_minus_1.shape)
    print(F0.shape)

    x = F0
    features = []
    for stage in stages:
        for block in stage:
            x = block(x)
        # Collect the output from the last block of each stage
        features.append(x)

    print(features[0].shape)
    print(features[1].shape)
