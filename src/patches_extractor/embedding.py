from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
from components.adaptive_maxvit_block.block_attention import BlockAttention
from components.adaptive_maxvit_block.mbconv_with_se import MBConvWithSE
from utils.utils import setup_logging, load_config
from components.sfeb import ShallowFeatureExtractionBlock
import logging

logger = logging.getLogger("my_application")


# Load an image and convert it to a tensor
def load_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


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


if __name__ == "__main__":
    # Call this at the start of your application to turn on/off logs
    setup_logging(os.path.join(os.getcwd(), "config", "logging_conf.yaml"))

    # Assuming an image path is specified
    image_path = (
        "/home/linuxu/Documents/datasets/div2k_train_pad_lr_bicubic_x4/0015.png"
    )
    image_tensor = load_image(image_path)

    # Initialize the PatchEmbedding module
    patch_embedding = PatchEmbedding(patch_size=64, emb_size=128, num_patches=64)

    # Move to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    patch_embedding = patch_embedding.to(device)

    # Apply the patch embedding module
    embedded_patches = patch_embedding(image_tensor)

    # Print out the shape of the output tensor
    print("Shape of embedded patches:", embedded_patches.shape)

    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]
    sfeb = ShallowFeatureExtractionBlock(config)
    sfeb = sfeb.to(device)
    F_minus1, F0 = sfeb(embedded_patches)
    print("Shape of f_minus_1 patches:", F_minus1.shape)
    print("Shape of f0 patches:", F0.shape)

    # Adaptive maxvit block
    mbconv_se = MBConvWithSE(
        in_channels=config["emb_size"], out_channels=config["emb_size"]
    )
    mbconv_se = mbconv_se.to(device)
    # Forward pass through MBConv with SE
    output = mbconv_se(F0, F0)
    print("MBOCNV with SE output shape:", output.shape)

    # ----------- Adaptive block attention ---------
    block_att = BlockAttention(
        dim=config["dim"],
        num_heads=config["num_heads"],
        block_size=config["block_size"],
    )
    block_att = block_att.to(device)
    bo = block_att(output)
