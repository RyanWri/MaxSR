import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader
from utils.plotting import show_patches


class PairedPatchesDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_patch_size=256, lr_patch_size=64):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size
        self.images = [
            img for img in os.listdir(lr_dir) if img.endswith(".png")
        ]  # Assume filenames match in both dirs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.images[idx])

        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_image = Image.open(lr_image_path).convert("RGB")

        # Extract patches
        lr_patches = self.extract_patches(lr_image, self.lr_patch_size)
        hr_patches = self.extract_patches(hr_image, self.hr_patch_size)

        return lr_patches, hr_patches

    def extract_patches(self, img, patch_size):
        img = transforms.ToTensor()(img)  # Convert to tensor
        patches = img.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
        return patches
