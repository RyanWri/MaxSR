import torchvision.transforms.functional as TF
from PIL import Image
from postprocessing.post_process import (
    visualize_feature_maps,
    visualize_attention_feature_maps,
    visualize_hffb_feature_maps,
    visualize_RB_output_image,
)
from components.sfeb import ShallowFeatureExtractionBlock
from components.adaptive_maxvit_block import AdaptiveMaxViTBlock
from components.hffb import HierarchicalFeatureFusionBlock
from components.reconstruction_block import ReconstructionBlock


# Assuming `image` is your input PIL Image
def process_image(image, out_channels):
    # Resize and possibly crop the image to 64x64
    image = TF.resize(image, (64, 64))
    image = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    visualize_RB_output_image(image, title="Input Image")

    # Initialize SFEB with 1 input channel (grayscale) or 3 (RGB), and some number of output channels
    sfeb = ShallowFeatureExtractionBlock(
        in_channels=image.shape[1], out_channels=out_channels
    )

    # Process the image through SFEB
    F0, F_minus_1 = sfeb(image)
    return F0, F_minus_1


if __name__ == "__main__":
    # Load an image
    img_path = "C:\Afeka\MaxSR\src\images\LR_bicubicx4.jpg"
    input_image = Image.open(img_path)

    # Process the image
    F0, F_minus_1 = process_image(input_image, out_channels=16)

    print("Output from SFEB:", F0.shape, F_minus_1.shape)

    # Visualize feature maps
    # visualize_feature_maps(F_minus_1)
    # visualize_feature_maps(F0)

    # Example instantiation and application
    adaptive_maxvit_block = AdaptiveMaxViTBlock(in_channels=16, dim=16)
    F0_adaptive = adaptive_maxvit_block(F0)  # F0 is the initial feature map from SFEB

    print("Shape of output after Adaptive MaxViT Block:", F0_adaptive.shape)

    # try to visualize adaptive maxvit block feature maps
    # visualize_attention_feature_maps(
    #     F0_adaptive, title="Output after Adaptive MaxViT Block"
    # )

    # Example of usage:
    # Assuming `features` is a list of feature maps from the last AMTB of each stage,
    # and `F_minus_1` is the output from the first convolution layer of the SFEB.
    features = [F0_adaptive]
    num_features = len(features)
    hffb = HierarchicalFeatureFusionBlock(channels=16, num_features=num_features)
    fused_features = hffb(features, F_minus_1)
    print("Shape of fused features:", fused_features.shape)

    # Visualize hierarchical feature fusion block feature maps
    # visualize_hffb_feature_maps(fused_features, title="Fused Features from HFFB")

    # Initialize the Reconstruction Block
    # Assuming the channel depth after pixel shuffle fits the final output requirement
    reconstruction_block = ReconstructionBlock(
        in_channels=16, out_channels=3, scale_factor=2
    )

    # Apply the Reconstruction Block
    reconstructed_image = reconstruction_block(fused_features)

    print("Shape of reconstructed image:", reconstructed_image.shape)

    # Visualize the reconstructed image
    visualize_RB_output_image(reconstructed_image)