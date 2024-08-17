import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch
from patches_extractor.embedding import embed_image


class LRHRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        """
        Args:
            lr_dir (string): Directory with all the low-resolution images.
            hr_dir (string): Directory with all the high-resolution images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = os.listdir(lr_dir)
        self.hr_images = os.listdir(hr_dir)
        self.transform = ToTensor()

        # Ensure that both folders have the same number of files and match each other
        assert len(self.lr_images) == len(
            self.hr_images
        ), "Mismatch in LR and HR dataset sizes"
        assert sorted(self.lr_images) == sorted(
            self.hr_images
        ), "LR and HR images do not match"

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_name = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_img_name = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_img_name).convert("RGB")
        hr_image = Image.open(hr_img_name).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


# Step 1: Define a custom dataset for precomputed embeddings and HR images
class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, model):
        # define transformer
        self.transform = ToTensor()
        # define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store HR imags from HR dir, transform to tensors
        # Convert LR images to tensors and store precompute embeddings
        self.hr_images = []
        self.embeddings = []

        # currently training on 100 images
        for image_name in os.listdir(hr_dir)[:100]:
            # HR images as tensors
            hr_image = Image.open(os.path.join(hr_dir, image_name)).convert("RGB")
            hr_image = self.transform(hr_image)
            self.hr_images.append(hr_image)

            # Embeddings
            lr_image_path = os.path.join(lr_dir, image_name)
            embeded_image = embed_image(lr_image_path, model, device, self.transform)
            self.embeddings.append(embeded_image)

        # Ensure that both folders have the same number of files and match each other
        assert len(self.embeddings) == len(
            self.hr_images
        ), "Mismatch in LR and HR dataset sizes"

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return a pair of precomputed embedding and HR image
        return self.embeddings[idx], self.hr_images[idx]
