""" 
implementation of maxSR according to MaxSr paper
Please refer to the paper to understand the architecture:
    https://arxiv.org/abs/2307.07240
"""

from layers.adaptive_maxvit_block import AdaptiveMaxViTBlock
from layers.sfeb import SFEB
from layers.hffb import HierarchicalFeatureFusionBlock
from layers.reconstruction_block import ReconstructionBlock
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MaxSR(nn.Module):
    def __init__(self, config: dict):
        super(MaxSR, self).__init__()
        self.sfeb = SFEB(
            config["in_channels"], config["sfeb_channels"], config["kernel_size"]
        )
        self.adaptive_maxvit_blocks = nn.ModuleList(
            [
                AdaptiveMaxViTBlock(
                    config["sfeb_channels"], config["num_heads"], config["ffn_dim"]
                )
                for _ in range(config["num_blocks"])
            ]
        )
        self.hffb = HierarchicalFeatureFusionBlock(config["sfeb_channels"])
        self.rb = ReconstructionBlock(
            config["sfeb_channels"], config["out_channels"], config["upscale_factor"]
        )
        self.output_blocks = []
        self.len_blocks = len(self.adaptive_maxvit_blocks)

    def forward(self, x):
        # SFEB Part
        sfeb_output = self.sfeb(x)
        F_minus1, F0 = sfeb_output
        features_map = [F0]

        # Cascaded Adaptive MaxVit Blocks
        block_output = F0
        for i, block in enumerate(self.adaptive_maxvit_blocks):
            logger.info(f"Running Block {i} / {self.len_blocks}")
            block_output = block(block_output)
            features_map.append(block_output)

        # HFFB Part
        hffb_output = self.hffb(F_minus1, features_map)

        # Reconstruction Block
        rb_output = self.rb(hffb_output)

        # Model image output is the output from the reconstruction block
        return rb_output
