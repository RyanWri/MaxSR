import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride_size=1,
        expand_rate=4,
        se_rate=0.25,
        dropout=0.0,
    ):
        super(MBConv, self).__init__()
        hidden_dim = int(expand_rate * in_dim)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride_size,
                kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.se = SE(hidden_dim, max(1, int(in_dim * se_rate)))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim)
        )

        self.proj = (
            nn.Identity()
            if in_dim == out_dim and stride_size == 1
            else nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim)
            )
        )

    def forward(self, x):
        residual = x
        logger.info(f"MBConv input shape: {x.shape}")
        x = self.expand_conv(x)
        x = self.dw_conv(x)
        logger.info(f"MBConv Expand and depthwise convolution completed")
        x = self.se(x)
        logger.info(f"MBConv squeeze and excitation completed")
        x = self.out_conv(x)
        if not isinstance(self.proj, nn.Identity):
            residual = self.proj(residual)
        mb_conv_se_output = x + residual
        logger.info(f"MBConv with SE output shape: {mb_conv_se_output.shape}")
        return mb_conv_se_output
