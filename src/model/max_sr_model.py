import torch.nn as nn
from components.sfeb import ShallowFeatureExtractionBlock
from components.amtb.amtb import AdaptiveMaxViTBlock
from components.hffb import HierarchicalFeatureFusionBlock
from components.reconstruction_block_super_tiny import ReconstructionBlockTiny


class MaxSRModel(nn.Module):
    def __init__(self, config):
        super(MaxSRModel, self).__init__()
        self.sfeb = ShallowFeatureExtractionBlock(config)
        blocks = tuple(
            AdaptiveMaxViTBlock(
                in_features=config["emb_size"],
                hidden_features=config["hidden_features"],
                out_features=config["dim"],
                heads=config["num_heads"],
            )
            for _ in range(config["block_per_stage"])
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*blocks) for _ in range(config["stages"])]
        )
        self.hffb = HierarchicalFeatureFusionBlock(
            num_stages=config["num_features"],
            in_features=config["emb_size"],
            out_features=config["emb_size"],
        )
        # Adjust scale_factor as needed
        self.reconstruction_block = ReconstructionBlockTiny(
            embedding_dim=config["dim"],
            patch_size=config["patch_size"],
            image_size=config["input_image_size"],
            scale_factor=config["scale_factor"],
        )

    def forward(self, x):
        # Process the image through SFEB
        F_minus_1, F0 = self.sfeb(x)
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
