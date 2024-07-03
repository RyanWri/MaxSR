import torch
import torch.nn as nn


class HFFB(nn.Module):
    def __init__(self, f_minus1, blocks_output: list):
        super(HFFB, self).__init__()
        self.blocks = blocks_output
        self.residual = f_minus1

        # Define convolutional layers to fuse the features from each stage
        self.fusion_layers = nn.ModuleList(
            [
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
                for _ in range(len(self.blocks))
            ]
        )

        # Final convolutional layer to combine all fused features
        self.final_conv = nn.Conv2d(
            num_channels * num_stages, num_channels, kernel_size=3, padding=1
        )

    def forward(self, stage_outputs):
        assert (
            len(stage_outputs) == self.num_stages
        ), "Number of stage outputs must match the number of stages"

        fused_features = []

        for i, stage_output in enumerate(stage_outputs):
            fused_feature = self.fusion_layers[i](stage_output)
            fused_features.append(fused_feature)

        # Concatenate all fused features along the channel dimension
        concatenated_features = torch.cat(fused_features, dim=1)

        # Apply the final convolution to combine the fused features
        output = self.final_conv(concatenated_features)

        return output
