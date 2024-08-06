import torch.nn as nn
from components.adaptive_maxvit_block.block_attention import BlockAttention
from components.adaptive_maxvit_block.grid_attention import GridAttention
from components.adaptive_maxvit_block.mb_conv_with_se import MBConvSE
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveMaxViTBlock(nn.Module):
    def __init__(self, config):
        super(AdaptiveMaxViTBlock, self).__init__()
        self.mbconv_se = MBConvSE(config["emb_size"], config["emb_size"])
        self.block_attention = BlockAttention(
            config["dim"], config["num_heads"], config["block_size"]
        )
        self.grid_attention = GridAttention(config["dim"], config["num_heads"])

    def forward(self, x):
        logger.info(f"AdaptiveMaxViT input shape: {x.shape}")
        x = self.mbconv_se(x)
        x = self.block_attention(x)
        x = self.grid_attention(x)
        logger.info(f"AdaptiveMaxViT output shape: {x.shape}")
        return x
