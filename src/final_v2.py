from model.maxsr import MaxSRModel
from postprocessing.post_process import visualize_RB_output_image
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block import AdaptiveMaxViTBlock
from utils.utils import load_config
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from preprossecing.img_datasets import PatchedDIV2KDataset, RescaleAndPatch
from torch.utils.data import DataLoader


# Image processing
def process_image(image_path, output_size):
    image = Image.open(image_path).convert("RGB")
    # Resize image using bicubic downsampling
    image = image.resize((output_size, output_size), Image.BICUBIC)
    return image


if __name__ == "__main__":
    config = load_config("C:\Afeka\MaxSR\src\config\maxsr_light.yaml")
    # Specify the directory containing your DIV2K images
    image_directory = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    transform = RescaleAndPatch(output_size=2048, downscale_factor=4, patch_size=64)
    dataset = PatchedDIV2KDataset(image_directory, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for i, patches in enumerate(data_loader):
        print("Batch", i, "Patch Shape:", patches.shape)
        if i > 5:  # Check first few batches to ensure it's working correctly
            break

        sfeb = ShallowFeatureExtractionBlock(config)
        sfeb.eval()

        stages = nn.ModuleList(
            [
                nn.Sequential(
                    AdaptiveMaxViTBlock(config),
                    AdaptiveMaxViTBlock(config),
                )
                for _ in range(2)  # Example: 2 stages, each with 2 blocks
            ]
        )
        # Forward pass
        with torch.no_grad():
            batch_size, patches_per_img, c, h, w = patches.size()
            patches = patches.view(
                -1, c, h, w
            )  # Flatten batch and patches dimensions for processing
            f0, f1 = sfeb(patches)
            x = f0
            features = []
            for stage in stages:
                for block in stage:
                    x = block(x)
                # Collect the output from the last block of each stage
                features.append(x)
    # # Instantiate and apply the complete model
    # maxsr_model = MaxSRModel()
    # input_image = process_image("C:\Afeka\MaxSR\src\images\LR_bicubicx4.jpg")
    # high_res_output = maxsr_model(input_image)

    # print("Shape of final high-resolution output:", high_res_output.shape)

    # # Visualize the reconstructed image
    # visualize_RB_output_image(high_res_output)
