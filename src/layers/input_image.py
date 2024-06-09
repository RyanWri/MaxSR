import torch
import torchvision.transforms as transforms
from PIL import Image


# Define a function to load and preprocess the image
def load_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert image to tensor
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


# Example usage
image_path = "path/to/your/image.jpg"
input_image = load_image(image_path)
