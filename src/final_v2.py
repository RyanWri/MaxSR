from model.maxsr import MaxSRModel
from postprocessing.post_process import visualize_RB_output_image
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block import AdaptiveMaxViTBlock
from utils.utils import load_config
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn


# Image processing
def process_image(image_path, output_size):
    image = Image.open(image_path).convert("RGB")
    # Resize image using bicubic downsampling
    image = image.resize((output_size, output_size), Image.BICUBIC)
    return image


if __name__ == "__main__":
    config = load_config("C:\Afeka\MaxSR\src\config\maxsr_light.yaml")
    image_path = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD/0001.png"
    processed_image = process_image(
        image_path, config["img_size"] // config["scale_factor"]
    )
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(processed_image).unsqueeze(0)

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
        f0, f1 = sfeb(input_tensor)
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
