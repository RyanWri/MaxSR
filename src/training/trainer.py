import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from torch import optim
from model.maxsr import MaxSRModel
import sys

print(sys.path)  # Check if the 'src' directory is listed in the path


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
            for img in os.listdir(image_dir)[:32]
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


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)

        # Assuming your model expects a low-resolution input and outputs a high-resolution image
        # You might need to create a low-res version of 'inputs' here if your model expects that

        optimizer.zero_grad()

        for image in inputs:
            outputs = model(image)
            loss = criterion(
                outputs, image
            )  # This assumes self-supervision; adjust as needed
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Save checkpoint after each epoch
        torch.save(
            model.state_dict(),
            os.path.join("C:\Machine Learning\models\MaxSR", f"model_epoch{i}.pth"),
        )

    return running_loss / len(dataloader)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    # Create the dataset
    train_dataset = DIV2KDataset(
        image_dir="C:\datasets\DIV2K\Dataset\DIV2K_train_LR_bicubic_X4\X4"
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming MaxSRModel has been defined and imported
    model = MaxSRModel().to(device)
    criterion = (
        torch.nn.MSELoss()
    )  # Common choice for regression tasks like super-resolution
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example of training for one epoch
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Training loss: {loss}")
