import torch
import torch.nn as nn


class HierarchicalFeatureFusionBlock(nn.Module):
    def __init__(self, num_stages, in_features, out_features):
        super(HierarchicalFeatureFusionBlock, self).__init__()
        # 1x1 Convolution to fuse features from different stages
        self.conv1x1 = nn.Conv2d(in_features * num_stages, out_features, kernel_size=1)

        # 3x3 Convolution to process the fused features
        self.conv3x3 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

    def forward(self, features, F_minus_1):
        """
        Args:
            features (list of Tensors): List of tensors from each stage. Each tensor has the shape [batch_size, num_patches, features].
            F_minus_1 (Tensor): The output from the first convolution in SFEB, shape [batch_size, num_patches, features].
        """
        # Concatenate features from all stages along the feature dimension
        x = torch.cat(features, dim=1)  # Concatenate along the feature dimension

        # Reshape to [batch_size, features, height, width] for 2D convolutions
        x = x.view(
            x.size(0), x.size(2) * len(features), 8, 8
        )  # Assuming num_patches is 64, reshape accordingly
        F_minus_1 = F_minus_1.view(
            F_minus_1.size(0), F_minus_1.size(2), 8, 8
        )  # Reshape F_minus_1 similarly

        # Apply 1x1 convolution to fuse the features
        x = self.conv1x1(x)

        # Apply 3x3 convolution to further process the fused features
        x = self.conv3x3(x)

        # Add the F_minus_1 tensor
        x = x + F_minus_1

        # Reshape back to the original format
        x = x.view(x.size(0), x.size(1), -1).permute(
            0, 2, 1
        )  # Reshape back to [batch_size, num_patches, features]

        return x
