from torchvision import transforms
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader


class ExtractPatches:
    def __init__(self, patch_size=64, stride=64):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, img):
        # img should be a tensor here
        patches = img.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        return patches.contiguous().view(
            -1, img.size(0), self.patch_size, self.patch_size
        )


class RescaleAndPatch:
    def __init__(self, output_size, downscale_factor=4, patch_size=64):
        self.output_size = output_size // downscale_factor
        self.patch_transform = ExtractPatches(patch_size)

    def __call__(self, img):
        img = transforms.Resize(
            (self.output_size, self.output_size), interpolation=Image.BICUBIC
        )(img)
        img = transforms.ToTensor()(img)
        img = self.patch_transform(img)
        return img


class PatchedDIV2KDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    # Use this in your dataset
    image_directory = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    transform = RescaleAndPatch(output_size=2048, downscale_factor=4, patch_size=64)
    dataset = PatchedDIV2KDataset(image_directory, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, patches in enumerate(data_loader):
        print("Batch", i, "Patch Shape:", patches.shape)
        if i > 2:  # Limit the output for testing
            break
