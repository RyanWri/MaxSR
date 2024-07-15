import torch
import torch.nn as nn


class HierarchicalFeatureFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(HierarchicalFeatureFusionBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, features, sfeb_first_conv_output):
        """
        features: list of features from the adaptive maxvit blocks
        sfeb_first_conv_output: output from the first convolution of the sfeb (F_-1)
        """
        out = self.conv1x1(features)
        out = self.conv3x3(out)
        return out + sfeb_first_conv_output
