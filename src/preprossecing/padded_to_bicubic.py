from PIL import Image
import os


def downscale_images(source_folder, target_folder, scale_factor=4):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Process each image in the source folder
    for filename in os.listdir(source_folder)[:1]:
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

    print(f"Images downscaled by factor of {scale_factor} and saved to {target_folder}")


if __name__ == "__main__":
    # Usage
    source_folder = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    target_folder = "C:\datasets\DIV2K\Dataset\DIV2K_train_LR_PAD_BICUBIC_x4"
    downscale_images(source_folder, target_folder)
