import torch
import torch.nn as nn
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HierarchicalFeatureFusionBlock(nn.Module):
    def __init__(self, channels, num_features):
        super(HierarchicalFeatureFusionBlock, self).__init__()
        self.concat_conv = nn.Conv2d(channels * num_features, channels, kernel_size=1)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, features, F_minus_1):
        logger.info(f"HFFB input shape: {len(features)}")
        # Concatenate features from different stages
        concatenated_features = torch.cat(features, dim=1)
        # Apply 1x1 convolution
        x = self.concat_conv(concatenated_features)
        # Apply 3x3 convolution
        x = self.final_conv(x)
        # Add the output of the first convolution layer of SFEB
        x += F_minus_1
        logger.info(f"HFFB output shape: {x.shape}")
        return x
