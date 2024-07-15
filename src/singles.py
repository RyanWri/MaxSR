from layers.mb_conv_with_se import MBConv
from preprossecing.input_image import load_image, preprocess_image
from layers.sfeb import SFEB
from layers.adaptive_maxvit_block import AdaptiveMaxViTBlock
import yaml
import logging
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_config(model_type: str) -> dict:
    if model_type == "light":
        config_path = "src/config/maxsr_light.yaml"
    elif model_type == "heavy":
        config_path = "src/config/maxsr_heavy.yaml"
    elif model_type == "test":
        config_path = "src/config/test.yaml"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file).get("maxsr_light")

    return config


if __name__ == "__main__":
    config = load_model_config(model_type="test")

    logger.info("Starting... LOADING IMAGE...")
    image_path = "C:\\datasets/DIV2K/Dataset/DIV2K_train_LR_bicubic/X2/0001x2.png"
    input_tensor = preprocess_image(image_path)
    logger.info(f"Image input Tensor shape : {input_tensor.shape}")
    logger.info("Completed... LOADING IMAGE...")

    logger.info("Starting... SFEB STAGE...")
    sfeb = SFEB(**config["SFEB"])
    sfeb_output = sfeb(input_tensor)
    f_minus_1, f_0 = sfeb_output
    logger.info("Completed... SFEB STAGE...")

    logger.info("Starting... ADAPTIVE MAXVIT BLOCKS STAGE...")
    logger.info("Starting... MBConv with Squeeze and Excitation STAGE...")
    amtb_config = config["AMTBs"]["block_settings"]
    mb_conv = MBConv(amtb_config["in_channels"], amtb_config["out_channels"])
    mb_conv_output = mb_conv(f_0)
    logger.info("completed... MBConv with Squeeze and Excitation STAGE...")
    logger.info("Completed... ADAPTIVE MAXVIT BLOCKS STAGE...")

    """
        TODO : Add Adaptive Block Attention (adaptive_block_sa + ffn)
        TODO : Add Adaptive Grid Attention (adaptive_grid_sa + ffn)
        TODO : Add HFFB (HierarchicalFeatureFusionBlock)
        TODO : Add ReconstructionBlock
        TODO : Plot the reconstructed image
    """


"""
    adaptive_maxvit_block = nn.ModuleList(
        [AdaptiveMaxViTBlock(config["AMTBs"]["block_settings"]) for i in range(1)]
    )
    amtbs_output = adaptive_maxvit_block(sfeb_output)
    
"""
