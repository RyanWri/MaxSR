from typing import Tuple
from utils.utils import calculate_np_mae_loss
import time


def process_single_image_patch(x_patch, y_patch, model, optimizer, criterion, device):
    """
    process single image
    """
    # Ensure the patch is on the correct device
    single_patch = x_patch.to(device)
    hr_single_patch = y_patch.to(device)

    # Clear existing gradients
    optimizer.zero_grad()
    # Process each patch through your model
    output = model(single_patch)

    # calculating loss corresponding to HR patch
    loss = criterion(output, hr_single_patch)
    loss.backward()
    optimizer.step()

    return loss.item()


def process_single_image(
    lr_patches, hr_patches, model, optimizer, criterion, device
) -> Tuple[float, float]:
    """
    process a single image splitted to 64 patches
    """
    losses = []
    start_time = time.time()
    # Assuming `lr_patch` is your tensor with shape (1, 64, 3, 64, 64)
    num_of_patches = lr_patches.shape[1]
    # Access the second dimension, which has 64 elements
    # Loop through each patch
    for patch_index in range(num_of_patches):
        x_patch = lr_patches[0, patch_index].unsqueeze(0)
        y_patch = hr_patches[0, patch_index].unsqueeze(0)

        loss = process_single_image_patch(
            x_patch=x_patch,
            y_patch=y_patch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        losses.append(loss)

    batch_loss = calculate_np_mae_loss(losses)
    batch_time = time.time() - start_time
    return batch_loss, batch_time


def process_batch(
    lr_patches, hr_patches, model, optimizer, criterion, device
) -> Tuple[float, float]:
    """
    in our case batch is a single image splitted to 64 patches
    """
    losses = []
    start_time = time.time()
    num_of_images = lr_patches.shape[0]
    for i in range(num_of_images):
        curr_lr_patches = lr_patches[i].unsqueeze(0)
        curr_hr_patches = hr_patches[i].unsqueeze(0)
        image_loss, image_time = process_single_image(
            lr_patches=curr_lr_patches,
            hr_patches=curr_hr_patches,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        losses.append(image_loss)

    batch_loss = calculate_np_mae_loss(losses)
    batch_time = time.time() - start_time
    return batch_loss, batch_time
