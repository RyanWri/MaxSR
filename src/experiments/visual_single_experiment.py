import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
from torchvision import transforms
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block.mb_conv_with_se import MBConvSE
from components.adaptive_maxvit_block.block_attention import BlockAttention
from components.adaptive_maxvit_block.grid_attention import GridAttention
from components.hffb import HierarchicalFeatureFusionBlock
from components.reconstruction_block import ReconstructionBlock

from postprocessing.post_process import (
    visualize_feature_maps, 
    visualize_mbconv_feature_maps, 
    visualize_attention_feature_maps,
    visualize_hffb_feature_maps, 
    visualize_RB_output_image
)

# Create the output directory
desktop_folder = os.path.expanduser("/Users/liav/Desktop/MaxSR_Liav")
os.makedirs(desktop_folder, exist_ok=True)

# Configuration
config = {
    "channels": 3,  # Number of channels in the input tensor
    "emb_size": 64  # Embedding size after convolution
}

# Load and preprocess the image
image_path = "/Users/liav/Downloads/liav.jpg"
image = Image.open(image_path).convert('RGB')

# Resize and transform the image to a tensor
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),        # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

x = transform(image).unsqueeze(0)  # Add batch dimension
# Initialize the shallow feature extraction block
sfeb = ShallowFeatureExtractionBlock(config)
#x = torch.rand(1, 3, 64, 64)  # Example input

# Pass through ShallowFeatureExtractionBlock
F_minus_1, F0 = sfeb(x)
save_path = os.path.join(desktop_folder, "F0.png")
visualize_feature_maps(F0, save_path)

# Initialize and apply MBConvSE block
in_channels = F0.size(1)
out_channels = 64  # Desired number of output channels for MBConvSE
mbconvse = MBConvSE(in_channels, out_channels)
mbconv_output = mbconvse(F0)
save_path = os.path.join(desktop_folder, "MBConvSE.png")
visualize_mbconv_feature_maps(mbconv_output, save_path)

# Initialize and apply BlockAttention
num_heads = 8  # Example number of attention heads
block_size = 4  # Example block size
block_attention = BlockAttention(out_channels, num_heads, block_size)
block_attention_output = block_attention(mbconv_output)
save_path = os.path.join(desktop_folder, "BlockAttention.png")
visualize_attention_feature_maps(block_attention_output, save_path, title="BlockAttention Feature Map")

# Initialize and apply GridAttention
grid_attention = GridAttention(out_channels, num_heads)  # Removed block_size
grid_attention_output = grid_attention(block_attention_output)
save_path = os.path.join(desktop_folder, "GridAttention.png")
visualize_attention_feature_maps(grid_attention_output, save_path, title="GridAttention Feature Map")

# Initialize and apply HierarchicalFeatureFusionBlock
num_features = 1  # Adjusted to the number of feature maps you are concatenating
hffb = HierarchicalFeatureFusionBlock(out_channels, num_features)
hffb_output = hffb([grid_attention_output], F_minus_1)  # Passing as a list to match expected input
save_path = os.path.join(desktop_folder, "HierarchicalFeatureFusionBlock.png")
visualize_hffb_feature_maps(hffb_output, save_path)

# Initialize and apply ReconstructionBlock
scale_factor = 2  # Example scale factor for upscaling
reconstruction_block = ReconstructionBlock(out_channels, config["channels"], scale_factor)
reconstruction_output = reconstruction_block(hffb_output)
save_path = os.path.join(desktop_folder, "ReconstructionBlock2.png")
visualize_RB_output_image(reconstruction_output, save_path)

print("end")