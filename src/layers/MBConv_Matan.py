import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

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
        
        # 1x1 Conv (Expand Convolution):
        # Expands the number of channels using a 1x1 convolution.
        # Followed by Batch Normalization and GELU activation.
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        # 3x3 Depthwise Conv:
        # Performs depthwise convolution with a 3x3 kernel, maintaining spatial dimensions and reducing computational complexity.
        # Followed by Batch Normalization and GELU activation.
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
        
        # Squeeze-and-Excitation Block:
        # Applies a global average pooling to generate channel-wise statistics.
        # Followed by two pointwise convolutions (1x1) and GELU activation to model channel-wise dependencies.
        # Finally, applies a sigmoid activation to scale the input.
        self.se = SE(hidden_dim, max(1, int(in_dim * se_rate)))
        
        # 1x1 Conv (Output Convolution):
        # Uses a 1x1 convolution to project the expanded channels back to the output dimension.
        # Followed by Batch Normalization.
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False), 
            nn.BatchNorm2d(out_dim)
        )
        
        # Residual Connection:
        # Ensures the input is directly added to the output.
        # If the input and output dimensions do not match, a 1x1 convolution with Batch Normalization is applied to the input to match dimensions.
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        ) if in_dim != out_dim or stride_size > 1 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.expand_conv(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.out_conv(x)
        if self.proj is not nn.Identity:
            residual = self.proj(residual)
        return x + residual
