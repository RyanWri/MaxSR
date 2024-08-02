import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from PIL import Image
import torchvision.transforms.functional as TF


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


def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)  # Convert to PIL Image

    # Define the preprocessing pipeline
    preprocess = Compose(
        [
            Resize((256, 256)),  # Resize to a fixed size
            ToTensor(),  # Convert to tensor
            Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    # Apply the preprocessing steps
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Assuming `image` is your input PIL Image
def process_image(image_path):
    image = Image.open(image_path)
    # Resize and possibly crop the image to 64x64
    image = TF.resize(image, (64, 64))
    image = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

    return image
