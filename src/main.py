import torch
import yaml
import os
from model.max_sr import MaxSR
from preprossecing.input_image import load_image
from postprocessing.post_process import postprocess_image


if __name__ == "__main__":
    model_type = "light"
    if model_type == "light":
        config_path = "src\config\maxsr_light.yaml"
    elif model_type == "heavy":
        config_path = "src\config\maxsr_heavy.yaml"
    elif model_type == "test":
        config_path = "src\config\test.yaml"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # build model with light or heavy params
    maxsr_model = MaxSR(config)

    # load data
    image_path = "images/LR_bicubicx4.jpg"
    input_tensor = load_image(image_path)

    # Run the model
    with torch.no_grad():
        output_tensor = maxsr_model(input_tensor)

    # save as image
    # Postprocess and display the output image
    output_image = postprocess_image(output_tensor)
    output_image.show()
