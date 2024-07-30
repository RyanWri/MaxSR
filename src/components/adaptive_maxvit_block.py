import torch.nn as nn
from components.block_attention import BlockAttention
from components.grid_attention import GridAttention
from components.mb_conv_with_se import MBConvSE
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, in_channels, dim, num_heads=4, block_size=8):
        super(AdaptiveMaxViTBlock, self).__init__()
        self.mbconv_se = MBConvSE(in_channels, dim)
        self.block_attention = BlockAttention(dim, num_heads, block_size)
        self.grid_attention = GridAttention(dim, num_heads)

    def forward(self, x):
        logger.info(f"AdaptiveMaxViT input shape: {x.shape}")
        x = self.mbconv_se(x)
        x = self.block_attention(x)
        x = self.grid_attention(x)
        logger.info(f"AdaptiveMaxViT output shape: {x.shape}")
        return x
