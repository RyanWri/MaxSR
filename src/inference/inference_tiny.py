import os
from patches_extractor.embedding import PatchEmbedding
from utils.plotting import save_tensor_as_image
from utils.utils import load_config
from model.max_sr_model import MaxSRModel
import torch
from torchvision import transforms
from PIL import Image


def setup_for_inference(model_path, device):
    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_super_tiny.yaml"))[
        "model_config"
    ]

    # Initialize the PatchEmbedding module
    patch_embedding_model = PatchEmbedding(
        patch_size=config["patch_size"],
        emb_size=config["emb_size"],
        num_patches=config["num_patches"],
    )
    # Embeding model
    patch_embedding = patch_embedding_model.to(device)

    # MaxSR model
    # Load the pre-trained weights onto the correct device
    model = MaxSRModel(config)
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    return model, patch_embedding


def do_inference(image_path, device):
    # open image
    lr_image = Image.open(image_path).convert("RGB")
    # transform to tensor
    transform = transforms.ToTensor()
    lr_image = transform(lr_image)
    lr_image = lr_image.to(device).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output1 = patch_embedding_model(lr_image)
        # use patch_embedding_model output as input to maxsr
        output2 = maxsr_model(output1)

    return output2


# Move to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup models
model_path = "/home/linuxu/Documents/models/MaxSR-Tiny/20240818_011948/model-checkpoints/model-epoch-34.pth"
maxsr_model, patch_embedding_model = setup_for_inference(model_path, device)
maxsr_model.eval()  # Set the model to inference mode
patch_embedding_model.eval()

# do inference
image_path = "/home/linuxu/Documents/datasets/Tiny_LR/4.jpeg"
output = do_inference(image_path, device)

# Process or print the final output
print(output.shape)
image_name = "/home/linuxu/Documents/model-output-images/reconstructed-tiny-4.jpeg"
save_tensor_as_image(tensor=output, file_path=image_name)
