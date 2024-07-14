from preprossecing.input_image import load_image, preprocess_image
from layers.sfeb import SFEB
import yaml

if __name__ == "__main__":
    model_type = "test"
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

    sfeb = SFEB(**config["SFEB"])

    image_path = "C:\\datasets/DIV2K/Dataset/DIV2K_train_LR_bicubic/X2/0001x2.png"

    input_tensor = preprocess_image(image_path)

    print(input_tensor.shape)

    output = sfeb(input_tensor)
