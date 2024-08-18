import os
from utils.plotting import save_tensor_as_image
from utils.utils import load_config, load_image
from model.maxsr_tiny import MaxSRTiny
import torch
from torchvision import transforms
from PIL import Image


def setup_for_inference(model_path, device):
    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_super_tiny.yaml"))[
        "model_config"
    ]

    # MaxSR model
    # Load the pre-trained weights onto the correct device
    model = MaxSRTiny()
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    return model


def do_inference(image_path, device):
    # Perform inference
    with torch.no_grad():
        image = load_image(image_path).to(device)
        output_image = maxsr_tiny_model(image)

    return output_image


# Move to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup models
model_path = "/home/linuxu/Documents/models/MaxSR-Tiny/20240818_172113/model-checkpoints/model-epoch-558.pth"
maxsr_tiny_model = setup_for_inference(model_path, device)
maxsr_tiny_model.eval()  # Set the model to inference mode

# do inference
image_path = "/home/linuxu/Documents/datasets/Tiny_LR/21.jpeg"
output = do_inference(image_path, device)

# Process or print the final output
print(output.shape)
image_name = (
    "/home/linuxu/Documents/model-output-images/MaxSR-Super-Tiny-output-21.jpeg"
)
save_tensor_as_image(tensor=output, file_path=image_name)
