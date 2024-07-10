""" 
implementation of maxSR according to MaxSr paper
Please refer to the paper to understand the architecture:
    https://arxiv.org/abs/2307.07240
"""

from layers.adaptive_maxvit_block import AdaptiveMaxViTBlock
from layers.sfeb import SFEB
from layers.hffb import HFFB
from layers.reconstruction_block import Reconstruction_block

import torch.nn as nn


class MaxSR(nn.Module):
    def __init__(self, in_channels, config):
        super(MaxSR, self).__init__()
        self.sfeb = SFEB()
        self.adaptive_maxvit_block = [AdaptiveMaxViTBlock(in_channels, config)] * 4
        self.output_blocks = []

    def forward(self, x):
        f_minus_1, x = self.sfeb(x)
        for block in self.adaptive_maxvit_block:
            x = block(x)
            self.output_blocks.append(x)

        x = HFFB(self.output_blocks) + f_minus_1

        image_output = Reconstruction_block(x)
        return image_output
