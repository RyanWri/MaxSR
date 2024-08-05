from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.images = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith(".png")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert("RGB")

        image = TF.resize(image, (64, 64))
        image = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

        return image
