import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# might need to add config and channels later
class SFEB(nn.Module):
    """
    Shallow Feature Extraction Block (SFEB) for extracting low-level features from the input image.
    The SFEB consists of two convolutional layers without activation functions.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SFEB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # the second layers gets it's input from the first layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)

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
