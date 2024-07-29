import torch.nn as nn


class HierarchicalFeatureFusionBlock(nn.Module):
    def __init__(self, channels, num_levels):
        super(HierarchicalFeatureFusionBlock, self).__init__()
        # Assumes features from `num_levels` different blocks or layers
        self.fusion_layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_levels)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *features):
        # Assume that features are passed in as a sequence of feature maps
        fused_feature = 0
        for feature, layer in zip(features, self.fusion_layers):
            fused_feature += layer(feature)
        return self.relu(fused_feature)
