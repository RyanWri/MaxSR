import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(
                channels, channels // reduction_ratio, 1
            ),  # Replace Linear with Conv2d
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels // reduction_ratio, channels, 1
            ),  # Replace Linear with Conv2d
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y  # Scale input by activation


class MBConvSE(nn.Module):
    def __init__(
        self, in_channels, out_channels, expansion_factor=1
    ):  # Reduced expansion factor to minimize complexity
        super(MBConvSE, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_res_connect = in_channels == out_channels

        self.expand_conv = (
            nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            if expansion_factor != 1
            else nn.Identity()
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.depthwise_conv = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.se = SEBlock(mid_channels)
        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x if self.use_res_connect else None
        x = self.expand_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.bn3(x)
        if self.use_res_connect:
            x += identity
        return x


if __name__ == "__main__":
    # Example usage
    input_tensor = torch.randn(1, 128, 64, 64)  # Example tensor
    model = MBConvSE(
        128, 128
    )  # Assuming the number of input and output channels are the same for simplicity
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
