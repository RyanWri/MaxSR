import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from model.maxsr import MaxSRModel
from preprossecing.img_datasets import DIV2KDataset


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
        image_dir="C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming MaxSRModel has been defined and imported
    model = MaxSRModel().to(device)
    criterion = (
        torch.nn.L1Loss()
    )  # Common choice for regression tasks like super-resolution
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example of training for one epoch
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Training loss: {loss}")
