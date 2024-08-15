import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


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
