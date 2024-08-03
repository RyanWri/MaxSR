import torchvision.transforms.functional as TF
from layers.sfeb import ShallowFeatureExtractionBlock
from components.mb_conv_with_se import MBConvSE
from components.block_attention import BlockAttention
from components.grid_attention import GridAttention
from PIL import Image
from postprocessing.post_process import (
    visualize_feature_maps,
    visualize_attention_feature_maps,
)


# Assuming `image` is your input PIL Image
def process_image(image, out_channels):
    # Resize and possibly crop the image to 64x64
    image = TF.resize(image, (64, 64))
    image = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

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

    # Example use of MBConvSE
    # Assume F0 is the output from the previous step (SFEB) with shape (batch_size, channels, height, width)
    mb_conv_se = MBConvSE(
        in_channels=16, out_channels=16
    )  # Adjust channels as per your SFEB output
    F0_se = mb_conv_se(F0)  # F0 is the output from SFEB

    print("Shape of output after MBConvSE:", F0_se.shape)

    # Example instantiation and application
    block_attention = BlockAttention(dim=16, num_heads=4, block_size=8)
    F0_attended = block_attention(F0_se)  # F0_se is the output from MBConvSE

    print("Shape of output after Block Attention:", F0_attended.shape)
    # try to visualize attention feature maps
    # visualize_attention_feature_maps(F0_attended, title="Output after Block Attention")

    # Example instantiation and application
    grid_attention = GridAttention(dim=16, num_heads=4)
    F0_grid_attended = grid_attention(
        F0_attended
    )  # Apply grid attention on the output from block attention

    print("Shape of output after Grid Attention:", F0_grid_attended.shape)

    # try to visualize grid attention feature maps
    visualize_attention_feature_maps(
        F0_grid_attended, title="Output after Grid Attention"
    )
