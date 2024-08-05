from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from utils.utils import extract_patches


class PatchedDIV2KDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, transform=None):
        self.image_dir = image_dir
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        patches = extract_patches(image, self.patch_size, self.patch_size)
        return patches
