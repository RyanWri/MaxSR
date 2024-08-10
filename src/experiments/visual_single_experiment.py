# TODO implement all stages

import torch
from components.sfeb import ShallowFeatureExtractionBlock
from postprocessing.post_process import visualize_feature_maps

config = {}
sfeb = ShallowFeatureExtractionBlock(config)
x = torch.rand(1, 3, 64, 64)
F_minus_1, F0 = sfeb(x)
visualize_feature_maps(F0)


# Input for first stage AdaptiveMaxViTBlock is F0
# x = F0
# features = []
# for stage in self.stages:
#     for block in stage:
#         x = block(x)
#     # Collect the output from the last block of each stage
#     features.append(x)
# x = self.hffb(features, F_minus_1)
# x = self.reconstruction_block(x)
# return x
