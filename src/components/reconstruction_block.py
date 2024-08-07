import torch.nn as nn
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(ReconstructionBlock, self).__init__()
        # Prepare for pixel shuffling
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # Followed by a 3x3 convolution
        self.final_conv = nn.Conv2d(
            in_channels // (scale_factor**2), out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        logger.info(f"ReconstructionBlock input shape: {x.shape}")
        x = self.pixel_shuffle(x)
        x = self.final_conv(x)
        logger.info(f"ReconstructionBlock output shape: {x.shape}")
        return x
