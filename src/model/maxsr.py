import torch.nn as nn
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block.adaptive_maxvit_block import AdaptiveMaxViTBlock
from components.hffb import HierarchicalFeatureFusionBlock
from components.reconstruction_block import ReconstructionBlock


class MaxSRModel(nn.Module):
    def __init__(self):
        super(MaxSRModel, self).__init__()
        self.sfeb = ShallowFeatureExtractionBlock(in_channels=3, out_channels=16)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    AdaptiveMaxViTBlock(in_channels=16, dim=16),
                    AdaptiveMaxViTBlock(in_channels=16, dim=16),
                )
                for _ in range(2)  # Example: 2 stages, each with 2 blocks
            ]
        )
        self.hffb = HierarchicalFeatureFusionBlock(channels=16, num_features=2)
        # Adjust scale_factor as needed
        self.reconstruction_block = ReconstructionBlock(
            in_channels=16, out_channels=3, scale_factor=1
        )

    def forward(self, x):
        # Process the image through SFEB
        F0, F_minus_1 = self.sfeb(x)
        # Input for first stage AdaptiveMaxViTBlock is F0
        x = F0
        features = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            # Collect the output from the last block of each stage
            features.append(x)
        x = self.hffb(features, F_minus_1)
        x = self.reconstruction_block(x)
        return x
