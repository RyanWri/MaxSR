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
        self.sfeb = SFEB(**config["SFEB"])
        self.num_blocks = config["AMTBs"]["num_blocks"]
        self.num_stages = config["AMTBs"]["num_stages"]
        self.adaptive_maxvit_blocks = nn.ModuleList(
            [
                AdaptiveMaxViTBlock(**config["AMTBs"]["block_settings"])
                for i in range(self.num_blocks)
            ]
        )
        self.hffb = HierarchicalFeatureFusionBlock(**config["HFFB"])
        self.rb = ReconstructionBlock(**config["RB"])
        self.output_blocks = []
        self.len_blocks = len(self.adaptive_maxvit_blocks)

    def forward(self, x):
        # SFEB Part
        sfeb_output = self.sfeb(x)
        F_minus1, F0 = sfeb_output
        features_map = [F0]

        # Cascaded Adaptive MaxVit Blocks
        block_output = F0
        blocks_per_stage = self.num_blocks // self.num_stages
        for stage in range(self.num_stages):
            logger.info(f"Running Stage {stage}")
            for i in range(blocks_per_stage):
                logger.info(f"Running Block {i} / {blocks_per_stage}")
                block_output = self.adaptive_maxvit_blocks[
                    (stage * blocks_per_stage) + i
                ](block_output)
                features_map.append(block_output)

            logger.info(f"Completed Stage {stage} ")

        # HFFB Part
        hffb_output = self.hffb(F_minus1, features_map)

        # Reconstruction Block
        rb_output = self.rb(hffb_output)

        # Model image output is the output from the reconstruction block
        return rb_output
