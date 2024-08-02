import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MBConvSE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=in_channels,
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Squeeze-and-Excitation layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(out_channels, out_channels // 16)
        self.se_fc2 = nn.Linear(out_channels // 16, out_channels)
        self.se_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Depthwise + Pointwise Convolution
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Squeeze-and-Excitation
        # Squeeze
        s = self.global_pool(x)
        s = torch.flatten(s, 1)
        s = F.relu(self.se_fc1(s))

        # Excitation
        s = self.se_sigmoid(self.se_fc2(s))
        s = s.view(-1, self.out_channels, 1, 1)

        # Scale the original features
        x = x * s
        return x
