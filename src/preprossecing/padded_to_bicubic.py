from PIL import Image
import os
import requests


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
    """
    relative to our server dataset folder -> /home/linuxu/Documents/datasets
    src_folder -> name of folder with HR images
    scale_factor -> int represent how much to reduce original image

    example usage:
        transform_padded_HR_images_to_LR_images(src_folder="div2k_validation_pad", scale_factor=4)

    output: None
        save all bicubic images in LR to target_dir
    """
    # our server root dataset folder
    base_dataset_path = "/home/linuxu/Documents/datasets"

    src_dir = f"{base_dataset_path}/{src_folder}"
    ouput_lr_dir = src_folder + f"_lr_bicubic_x{str(scale_factor)}"
    target_dir = f"{base_dataset_path}/{ouput_lr_dir}"

    downscale_images(src_dir, target_dir, scale_factor=4)


def generate_random_image():
    for i in range(100):
        resp = requests.get("https://picsum.photos/128")
        if resp.ok:
            with open(
                f"/home/linuxu/Documents/datasets/Tiny_HR/{i}.jpeg", mode="wb"
            ) as fp:
                fp.write(resp.content)


if __name__ == "__main__":
    downscale_images(
        source_folder="/home/linuxu/Documents/datasets/Tiny_HR",
        target_folder="/home/linuxu/Documents/datasets/Tiny_LR",
        scale_factor=2,
    )
