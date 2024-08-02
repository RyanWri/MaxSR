from PIL import Image
from preprossecing.input_image import process_image
import torch
from model.maxsr import MaxSRModel
import matplotlib.pyplot as plt
from postprocessing.post_process import imshow, visualize_RB_output_image

if __name__ == "__main__":
    # Assuming MaxSRModel is already defined and imported
    model = MaxSRModel()  # Instantiate the model
    model.load_state_dict(
        torch.load("C:\Machine Learning\models\MaxSR\model_base.pth")
    )  # Load the saved weights
    model.eval()  # Set the model to evaluation mode

    # Preprocess the input image
    input_img = process_image(
        "C:\datasets\DIV2K\Dataset\DIV2K_train_LR_bicubic_X4\X4/0010x4.png"
    )

    output_img = model(input_img)  # Pass the image through the model

    # Display images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    visualize_RB_output_image(input_img, "Input Image")
    plt.subplot(1, 2, 2)
    visualize_RB_output_image(output_img, "Super-Resolved Output")
    plt.show()
