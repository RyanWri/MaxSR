from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import logging
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("my_application")


# Load an image and convert it to a tensor
def load_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# precomputed embeddings for a single image with 64 patches to a representating vector
def embed_image(image_absolute_path, model, device, transform):
    # open image
    lr_image = Image.open(image_absolute_path).convert("RGB")
    # transform to tensor
    lr_image = transform(lr_image)
    lr_image = lr_image.to(device).unsqueeze(
        0
    )  # Move to device and add batch dimension
    # Compute the embedding using the model and return to cpu
    with torch.no_grad():  # Disable gradient calculation for inference
        embedded_tensor = model(lr_image).detach().cpu()
    # Remove the batch dimension and return the tensor
    return embedded_tensor.squeeze(0)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, emb_size, num_patches):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_patches = num_patches

        # Linear projection of flattened patches
        self.patch_to_emb = nn.Linear(3 * patch_size * patch_size, emb_size)

        # Learnable positional encodings
        self.pos_embeddings = nn.Parameter(torch.randn(num_patches, emb_size))

    def forward(self, x):
        """
        Args:
        x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
        Tensor: Output tensor with embedded and position-encoded patches
        """
        # x shape: (batch_size, channels, height, width)
        # Create patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size)  # Create patches along height
        x = x.unfold(3, self.patch_size, self.patch_size)  # Create patches along width
        x = x.contiguous().view(
            x.size(0), -1, 3 * self.patch_size * self.patch_size
        )  # Flatten patches

        # Apply linear projection to each patch
        x = self.patch_to_emb(x)  # shape: (batch_size, num_patches, emb_size)

        # Add positional encodings
        x += self.pos_embeddings.unsqueeze(0)  # Broadcasting over the batch size

        return x


class SineCosinePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, num_patches):
        super(SineCosinePositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        # Create the sine-cosine positional encodings
        position = np.arange(self.num_patches)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2)
            * -(np.log(10000.0) / self.embedding_dim)
        )
        pos_enc = np.zeros((self.num_patches, self.embedding_dim))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return torch.from_numpy(pos_enc).float().unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x shape: (batch_size, num_patches, embedding_dim)
        return x + self.positional_encoding.to(x.device)


class ImageEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim, num_patches):
        super(ImageEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_size * patch_size * 3, embedding_dim)
        self.positional_encoding = SineCosinePositionalEncoding(
            embedding_dim, num_patches
        )

    def forward(self, x):
        # Flatten and project patches
        x = x.view(
            x.size(0), x.size(1), -1
        )  # (batch_size, num_patches, patch_size * patch_size * 3)
        x = self.projection(x)  # Linear projection (embedding)

        # Add positional encodings
        x = self.positional_encoding(x)

        return x


def embed_image_sin_cosine(image_absolute_path, model, patch_size, device):
    """
    Embed an RGB image into patches, apply linear projection, and add positional encodings.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        model (nn.Module): Vision Transformer model with embedding and positional encoding layers.
        patch_size (int): The size of each patch (default: 8).
        embedding_dim (int): The dimension of the embedding space (default: 256).

    Returns:
        torch.Tensor: Embedded image patches with positional encodings, shape (1, num_patches, embedding_dim).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    lr_image = Image.open(image_absolute_path).convert("RGB")
    lr_image = transform(lr_image)
    image = lr_image.unsqueeze(0).to(device)

    # Unfold the image into patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(
        1, 3, -1, patch_size, patch_size
    )  # (1, 3, num_patches, patch_size, patch_size)
    patches = patches.permute(
        0, 2, 1, 3, 4
    )  # (1, num_patches, 3, patch_size, patch_size)
    patches = patches.contiguous().view(
        1, patches.size(1), -1
    )  # (1, num_patches, 3 * patch_size * patch_size)

    # Pass the patches through the model (embedding + positional encoding)
    embedded_patches = model(patches).detach().cpu()
    return embedded_patches.squeeze(0)
