import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(),  # Activation here is typical to introduce non-linearity in the SE block
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBConvWithSE, self).__init__()
        self.expand_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.depthwise_conv = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels
        )
        self.se_block = SqueezeExcitation(out_channels)
        self.project_conv = nn.Conv2d(out_channels, in_channels, 1)

    def forward(self, x, F0):
        # Save input for residual connection
        residual = F0

        # 1x1 convolution - expansion
        x = self.expand_conv(x)

        # 3x3 depthwise convolution
        x = self.depthwise_conv(x)

        # Squeeze-and-Excitation block
        x = self.se_block(x)

        # 1x1 convolution - projection
        x = self.project_conv(x)

        # Adding the residual (F0)
        x = x + residual

        return x
