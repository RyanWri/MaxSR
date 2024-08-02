import torch
import torch.nn as nn
from src.layers.adaptive_block_self_attention import AdaptiveBlockSelfAttention
from src.layers.ffn import FFN
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveBlockAttention(nn.Module):
    def __init__(self, config):
        super(AdaptiveBlockAttention, self).__init__()
        # Initialize the Self-Attention component
        self.adaptive_block_sa = AdaptiveBlockSelfAttention(config)

        # Initialize the Feed-Forward Network component
        self.ffn = FFN(config)

    def forward(self, x):
        logger.info("Starting... Adaptive Block Attention STAGE...")
        # Apply self-attention
        x = self.adaptive_block_sa(x)
        logger.info("Completed... Adaptive Block Attention STAGE...")
        # Apply Feed-Forward Network
        logger.info("Starting... FFN STAGE...")
        x = self.ffn(x)
        logger.info("Completed... FFN STAGE...")

        return x
