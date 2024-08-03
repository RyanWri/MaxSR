from model.maxsr import MaxSRModel
from preprossecing.input_image import process_image
from postprocessing.post_process import visualize_RB_output_image


# Instantiate and apply the complete model
maxsr_model = MaxSRModel()
input_image = process_image("C:\Afeka\MaxSR\src\images\LR_bicubicx4.jpg")
high_res_output = maxsr_model(input_image)

print("Shape of final high-resolution output:", high_res_output.shape)

# Visualize the reconstructed image
visualize_RB_output_image(high_res_output)
