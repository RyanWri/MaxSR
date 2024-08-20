import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Define the Shallow Feature Extractor (SFE) with two 3x3 convolutions
class ShallowFeatureExtractor(nn.Module):
    def __init__(self):
        super(ShallowFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        return x


# Function to extract patches from the feature map
def extract_patches(feature_map, patch_size):
    batch_size, channels, height, width = feature_map.size()
    patches = feature_map.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.contiguous().view(
        batch_size, channels, -1, patch_size, patch_size
    )
    return patches


# Class for learned positional embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, x):
        return x + self.pos_embedding


# MBConv Block with Squeeze and Excitation
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, se_ratio=0.25):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.se_reduce = nn.Conv2d(
            hidden_dim, int(in_channels * se_ratio), kernel_size=1
        )
        self.se_expand = nn.Conv2d(
            int(in_channels * se_ratio), hidden_dim, kernel_size=1
        )
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Expansion
        out = self.expand_conv(x)
        out = F.relu(self.norm1(out))

        # Depthwise Convolution
        out = self.depthwise_conv(out)
        out = F.relu(self.norm2(out))

        # Squeeze and Excitation
        se = F.adaptive_avg_pool2d(out, 1)
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        out = out * se

        # Projection
        out = self.project_conv(out)
        out = self.norm3(out)

        return out


# Block Attention (Multi-head Self-Attention)
class BlockAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(BlockAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Switch to (sequence, batch, embedding) for attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # Switch back to (batch, sequence, embedding)
        return x


# Grid Attention (Another Multi-head Self-Attention)
class GridAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GridAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Switch to (sequence, batch, embedding) for attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # Switch back to (batch, sequence, embedding)
        return x


# Adaptive MaxViT Block combining MBConv, Block Attention, and Grid Attention
class AdaptiveMaxViTBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, expansion_factor=4, se_ratio=0.25, mlp_dim=256
    ):
        super(AdaptiveMaxViTBlock, self).__init__()
        self.mbconv = MBConv(embed_dim, embed_dim, expansion_factor, se_ratio)
        self.block_attn = BlockAttention(embed_dim, num_heads)
        self.grid_attn = GridAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape the input to fit the MBConv block
        batch_size, num_patches, embed_dim = x.shape
        height = width = int(num_patches**0.5)
        x = x.view(batch_size, height, width, embed_dim).permute(
            0, 3, 1, 2
        )  # Reshape to (batch, embed_dim, height, width)

        # MBConv with Squeeze and Excitation
        x = self.mbconv(x)

        # Reshape back to (batch_size, num_patches, embed_dim)
        x = x.view(batch_size, embed_dim, -1).permute(0, 2, 1)

        # Block Attention
        x = self.block_attn(x)

        # Grid Attention
        x = self.grid_attn(x)

        # Feed-Forward Network (MLP Block)
        mlp_output = self.mlp(x)
        x = self.norm(x + mlp_output)

        return x


# Hierarchical Feature Fusion Block
class HFFB(nn.Module):
    def __init__(self, embed_dim, num_stages):
        super(HFFB, self).__init__()
        self.convs_1x1 = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(num_stages)]
        )
        self.convs_3x3 = nn.ModuleList(
            [
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
                for _ in range(num_stages)
            ]
        )
        self.fusion_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, features):
        fused_features = 0
        for i, feature in enumerate(features):
            # Reshape the feature to 2D convolutional format
            batch_size, num_patches, embed_dim = feature.shape
            height = width = int(num_patches**0.5)
            feature = feature.view(batch_size, height, width, embed_dim).permute(
                0, 3, 1, 2
            )  # Shape: (batch, embed_dim, height, width)

            feature = self.convs_1x1[i](feature)  # 1x1 convolution to reduce channels
            feature = self.convs_3x3[i](
                feature
            )  # 3x3 convolution for spatial refinement
            fused_features += feature

        fused_features = self.fusion_conv(
            fused_features
        )  # Final fusion with another 3x3 convolution

        # Reshape back to (batch_size, num_patches, embed_dim)
        fused_features = fused_features.permute(0, 2, 3, 1).view(
            batch_size, -1, embed_dim
        )
        return fused_features


# Reconstruction Block
class ReconstructionBlock(nn.Module):
    def __init__(
        self, embed_dim, num_patches, patch_size, out_channels, upscale_factor=2
    ):
        super(ReconstructionBlock, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        # Linear layer to project the embedded patches back to image space
        self.projection = nn.Linear(
            embed_dim, (out_channels * (upscale_factor**2)) * patch_size * patch_size
        )

        # Pixel shuffle for upscaling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Final 3x3 convolution
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # Project patches back to image space
        x = self.projection(
            x
        )  # Shape: (batch_size, num_patches, (out_channels * upscale_factor^2) * patch_size * patch_size)
        x = x.view(
            batch_size,
            num_patches,
            self.out_channels * (self.upscale_factor**2),
            self.patch_size,
            self.patch_size,
        )  # (batch_size, num_patches, out_channels * upscale_factor^2, patch_size, patch_size)

        # Combine patches into a full image
        height = width = int(num_patches**0.5)
        x = x.permute(
            0, 2, 1, 3, 4
        )  # Shape: (batch_size, out_channels * upscale_factor^2, num_patches, patch_size, patch_size)
        x = x.contiguous().view(
            batch_size,
            self.out_channels * (self.upscale_factor**2),
            height * self.patch_size,
            width * self.patch_size,
        )  # (batch_size, out_channels * upscale_factor^2, height*patch_size, width*patch_size)

        # Apply pixel shuffle to rearrange to the desired spatial size
        x = self.pixel_shuffle(
            x
        )  # Upscales and reduces channels to match the desired output (batch_size, out_channels, height*upscale_factor, width*upscale_factor)

        # Apply final convolution to refine the output
        x = self.conv(
            x
        )  # Final output (batch_size, out_channels, height*upscale_factor, width*upscale_factor)

        # Adjust final resolution to (128, 128)
        if x.shape[2] > 128:
            x = nn.functional.interpolate(
                x, size=(128, 128), mode="bilinear", align_corners=False
            )

        return x


# Define the MaxSR Model
class MaxSRSuperTiny(nn.Module):
    def __init__(self, config):
        super(MaxSRSuperTiny, self).__init__()
        self.patch_size = config["patch_size"]
        self.embed_dim = config["emb_size"]
        self.num_heads = config["num_heads"]
        self.mlp_dim = config["dim"]
        self.expansion_factor = 4
        self.se_ratio = 0.25
        self.sfe = ShallowFeatureExtractor()
        # Patch embedding layer
        self.patch_embeddings = nn.Linear(
            64 * self.patch_size * self.patch_size, self.embed_dim
        )

        # Positional embedding layer
        self.num_patches = config["num_patches"]
        self.positional_embedding = PositionalEmbedding(
            self.num_patches, self.embed_dim
        )

        # Adaptive MaxViT Blocks
        blocks = tuple(
            AdaptiveMaxViTBlock(
                self.embed_dim,
                self.num_heads,
                self.expansion_factor,
                self.se_ratio,
                self.mlp_dim,
            )
            for _ in range(config["block_per_stage"])
        )
        self.stages = nn.ModuleList(
            [nn.Sequential(*blocks) for _ in range(config["stages"])]
        )

        self.hffb = HFFB(embed_dim=self.embed_dim, num_stages=config["stages"])

        # Reconstruction block
        self.reconstruction = ReconstructionBlock(
            self.embed_dim, self.num_patches, self.patch_size, 3
        )

    def forward(self, x):
        # Step 1: Shallow Feature Extraction
        feature_map = self.sfe(x)

        # Step 2: Extract Patches
        patches = extract_patches(feature_map, self.patch_size)

        # Step 3: Flatten each patch and project to a fixed dimension
        batch_size, channels, num_patches, _, _ = patches.shape
        patches = patches.view(
            batch_size, channels, num_patches, -1
        )  # Flatten each patch
        patches = patches.permute(
            0, 2, 1, 3
        ).contiguous()  # Move channels to the right position
        patches = patches.view(
            batch_size, num_patches, -1
        )  # Final shape: (batch_size, num_patches, channels * patch_size^2)
        patches = self.patch_embeddings(patches)  # Project to fixed dimension

        # Step 4: Add Learned Positional Embeddings
        patches = self.positional_embedding(patches)

        # Step 5: Process through Adaptive MaxViT Block (Transformer Encoder)
        # Step 5: Process through Adaptive MaxViT Block (Transformer Encoder)
        x = patches
        features = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            # Collect the output from the last block of each stage
            features.append(x)

        # Step 6: Process through HFFB
        fused_features = self.hffb(features)

        # Step 7: Process through Reconstruction Block
        output_image = self.reconstruction(fused_features)

        return output_image
