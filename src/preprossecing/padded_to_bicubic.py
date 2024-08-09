from PIL import Image
import os


def downscale_images(source_folder, target_folder, scale_factor=4):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Process each image in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Open the image
            with Image.open(source_path) as img:
                # Compute the new dimensions
                new_width = img.width // scale_factor
                new_height = img.height // scale_factor

                # Resize the image using bicubic interpolation
                img_resized = img.resize((new_width, new_height), Image.BICUBIC)

                # Save the resized image to the target folder
                img_resized.save(target_path)


def transform_padded_HR_images_to_LR_images(src_folder: str, scale_factor: int) -> None:
    # our server root dataset folder
    base_dataset_path = "/home/linuxu/Documents/datasets"

    src_dir = f"{base_dataset_path}/{src_folder}"
    ouput_lr_dir = src_folder + f"_lr_bicubic_x{str(scale_factor)}"
    target_dir = f"{base_dataset_path}/{ouput_lr_dir}"

    downscale_images(src_dir, target_dir, scale_factor=4)

if __name__ == "__main__":
    transform_padded_HR_images_to_LR_images(src_folder="div2k_validation_pad", scale_factor=4)
