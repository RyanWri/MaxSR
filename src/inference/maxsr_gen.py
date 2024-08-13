import torch
import torch.nn as nn
import os
from utils.utils import load_config, setup_logging
from utils.plotting import plot_image
from model.maxsr import MaxSRModel
from inference.processor import inference_image_to_patches, reassemble_patches
from torchvision.transforms.functional import to_pil_image


if __name__ == "__main__":
    # Call this at the start of your application to turn on/off logs
    setup_logging(os.path.join(os.getcwd(), "config", "logging_conf.yaml"))

    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]

    # Assuming the model class and path are correctly defined
    model = MaxSRModel(config)
    model_path = "/home/linuxu/Documents/models/MaxSR/20240810_192641/version-0-0-5.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    image_name = "0028.png"
    test_image = (
        f"/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4/{image_name}"
    )
    input_patches = inference_image_to_patches(image_path=test_image)
    output_patches = []
    with torch.no_grad():
        for patch in input_patches:
            patch = patch.unsqueeze(0).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )  # Add batch dimension
            output = model(patch)
            output_patches.append(output.squeeze())

    # Assuming 'patches' is your list of 64 tensors, each of shape (3, 256, 256)
    patches_tensor = torch.stack(output_patches)

    # Now 'patches_tensor' should have the shape (64, 3, 256, 256)
    print("Shape of the combined tensor:", patches_tensor.shape)
    # Assuming `output_patches` is a tensor with shape [64, 3, 256, 256] where 64 patches are lined up
    output_image = reassemble_patches(patches_tensor, patches_per_row=8)
    plot_image(output_image, filename=image_name)
