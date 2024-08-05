import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ShallowFeatureExtractionBlock(nn.Module):
    def __init__(self, config):
        super(ShallowFeatureExtractionBlock, self).__init__()
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(
            config["channels"], config["emb_size"], kernel_size=3, stride=1, padding=1
        )
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(
            config["emb_size"], config["emb_size"], kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        Forward pass for the SFEB.

        Args:
        x (torch.Tensor): Input image tensor.

        Returns:
        torch.Tensor: Feature map after the first convolutional layer (F_minus_1).
        torch.Tensor: Feature map after the second convolutional layer (F0).
        """
        # print shape of input tensor (x.shape)
        logger.info(f"SFEB input shape: {x.shape}")
        F_minus_1 = self.conv1(x)
        logger.info(f"SFEB First conv output (F_minus_1) shape: {F_minus_1.shape}")
        F0 = self.conv2(F_minus_1)
        logger.info(f"SFEB Second conv output (F0) shape: {F0.shape}")
        return F_minus_1, F0
