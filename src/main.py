import torch
import torch.nn as nn
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block.mbconv_with_se import MBConvSE
from components.adaptive_maxvit_block.block_attention import BlockAttention
from components.adaptive_maxvit_block.grid_attention import GridAttention
from components.adaptive_maxvit_block.adaptive_maxvit_block import AdaptiveMaxViTBlock
from components.hffb import HierarchicalFeatureFusionBlock
from utils.utils import load_config


def test_inside_maxvit_block(F0, config):
    mconv = MBConvSE(config["emb_size"], config["emb_size"])
    x = mconv(F0)
    print("Mbconv with SE Output Shape:", x.shape)

    block_attention = BlockAttention(
        config["dim"], config["num_heads"], config["block_size"]
    )
    x = block_attention(x)
    print("Block Attention Output Shape:", x.shape)

    print("Total number of parameters:", sum(p.numel() for p in sfeb.parameters()))

    grid_attention = GridAttention(config["dim"], config["num_heads"])
    x = grid_attention(x)
    print("Grid Attention Output Shape:", x.shape)

    adaptive_block = AdaptiveMaxViTBlock(config)
    y = adaptive_block(F0)
    print("Adaptive MaxViT Output Shape:", y.shape)


if __name__ == "__main__":
    # Example usage
    input_patch = torch.randn(1, 3, 64, 64)  # Batch size, Channels, Height, Width
    # Load configuration
    config = load_config("C:\Afeka\MaxSR\src\config\maxsr_tiny.yaml")["model_config"]

    sfeb = ShallowFeatureExtractionBlock(config)
    print("Input Shape:", input_patch.shape)
    F_minus_1, F0 = sfeb(input_patch)
    print("F0 Shape:", F0.shape)
    print("F_minus_1 Shape:", F_minus_1.shape)

    x = F0
    stages = nn.ModuleList(
        [
            nn.Sequential(
                AdaptiveMaxViTBlock(config),
                AdaptiveMaxViTBlock(config),
            )
            for _ in range(2)  # Example: 2 stages, each with 2 blocks
        ]
    )

    features = []
    for i, stage in enumerate(stages):
        for block in stage:
            x = block(x)
        print(f"Stage {i} Output Shape:", x.shape)
        # Collect the output from the last block of each stage
        features.append(x)

    print("stages complete")
    hffb = HierarchicalFeatureFusionBlock(config["emb_size"], config["num_features"])
    x = hffb(features, F_minus_1)
    print("HFBF Output Shape:", x.shape)
