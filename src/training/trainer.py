import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from model.maxsr import MaxSRModel
from preprossecing.img_datasets import DIV2KDataset


def train(model, dataloader, optimizer, criterion, device):
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

    return running_loss / len(dataloader)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
