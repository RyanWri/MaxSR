import os
import shutil
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def transform_image(image, rotation_angle=0, flip_horizontal=False, flip_vertical=False, add_noise=False):
    """
    Applies transformations to the image such as rotation, flipping, and adding noise.
    :param image: PIL Image object.
    :param rotation_angle: Angle to rotate the image.
    :param flip_horizontal: Boolean to flip the image horizontally.
    :param flip_vertical: Boolean to flip the image vertically.
    :param add_noise: Boolean to add Gaussian noise to the image.
    :return: Transformed PIL Image object.
    """
    #print(f"Applying transformations: rotation_angle={rotation_angle}, flip_horizontal={flip_horizontal}, flip_vertical={flip_vertical}, add_noise={add_noise}")
    # Rotate the image
    if rotation_angle != 0:
        image = image.rotate(rotation_angle, expand=True)
    
    # Flip the image horizontally
    if flip_horizontal:
        image = ImageOps.mirror(image)
    
    # Flip the image vertically
    if flip_vertical:
        image = ImageOps.flip(image)
    
    # Add Gaussian noise
    if add_noise:
        noise = Image.effect_noise(image.size, random.uniform(5, 10))
        image = Image.blend(image, noise.convert("RGB"), 0.2)
    
    return image

def augment_and_save_images(input_folder_path, output_folder_path, augmentations=3):
    """
    Applies various transformations to the images in the input folder and saves the results to the output folder.
    :param input_folder_path: Path to the folder containing original image datasets.
    :param output_folder_path: Path to the folder where augmented images will be saved.
    :param augmentations: Number of augmented versions to create for each image.
    """
    print(f"Starting augmentation process for images in {input_folder_path}, saving to {output_folder_path}")
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")

    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))])
    print(f"Found {len(image_files)} images in the input folder.")


    # Transform and save the images
    for filename in image_files:
        input_image_path = os.path.join(input_folder_path, filename)
        
        # Open the input image
        try:
            image = Image.open(input_image_path)
            print(f"Processing image: {filename}")
            
            # Apply transformations and save multiple augmented versions
            for i in range(augmentations):
                # Define random transformations
                rotation_angle = random.choice([0, 90, 180, 270])
                flip_horizontal = random.choice([True, False])
                flip_vertical = random.choice([True, False])
                add_noise = random.choice([True, False])
                
                # Apply transformations
                transformed_image = transform_image(image, rotation_angle, flip_horizontal, flip_vertical, add_noise)
                
                # Save the transformed image to the output folder
                base_name, ext = os.path.splitext(filename)
                output_image_path = os.path.join(output_folder_path, f"{base_name}_aug_{i + 1}{ext}")
                transformed_image.save(output_image_path)
                print(f"Saved augmented image: {output_image_path}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# # Example usage:
# input_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/Div2k"
# output_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/Div2k_scaled"
# #augment_and_save_images(input_folder, output_folder, augmentations=3)

def duplicate_and_rename_images(folder_path, new_folder_path):
    """
    Duplicates the folder at folder_path to new_folder_path and renames the images in the duplicated folder.
    :param folder_path: Path to the original folder containing image datasets.
    :param new_folder_path: Path to the duplicated folder.
    """
    print(f"Duplicating images from {folder_path} to {new_folder_path}")
    # Check if the new folder already exists
    if os.path.exists(new_folder_path):
        # Remove the existing folder
        shutil.rmtree(new_folder_path)
        print(f"Removed existing folder: {new_folder_path}")

    # Duplicate the folder
    shutil.copytree(folder_path, new_folder_path)
    print(f"Duplicated folder from {folder_path} to {new_folder_path}")
    
    # Get the list of image files in the new folder and sort them
    image_files = sorted([f for f in os.listdir(new_folder_path) if os.path.isfile(os.path.join(new_folder_path, f))])
    print(f"Found {len(image_files)} images in the new folder.")

    # Rename the images
    for idx, filename in enumerate(image_files):
        old_path = os.path.join(new_folder_path, filename)
        new_filename = f"div2k_HR_{idx + 1}.jpg"
        new_path = os.path.join(new_folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

# # Example usage:
# original_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/Div2k_scaled"
# duplicated_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/Div2k_HR"
# print("Started duplicating and renaming images")
# #duplicate_and_rename_images(original_folder, duplicated_folder)
# print("Done duplicating and renaming images")

def process_image(image, scale_factor=4):
    """
    Downscales the image using bicubic interpolation.
    :param image: PIL Image object.
    :param scale_factor: Factor by which to downscale the image.
    :return: Downscaled PIL Image object.
    """
    original_size = image.size  # (width, height)
    target_size = (original_size[0] // scale_factor, original_size[1] // scale_factor)
    print(f"Downscaling image from size {original_size} to {target_size}")
    
    # Downscale the image using bicubic interpolation
    downscaled_image = image.resize(target_size, Image.BICUBIC)
    
    return downscaled_image

def process_and_save_images(hr_folder_path, lr_folder_path, scale_factor=4):
    """
    Processes (downscales) the images in the HR folder and saves the results to the LR folder
    with the same filenames.
    :param hr_folder_path: Path to the HR folder containing high-resolution image datasets.
    :param lr_folder_path: Path to the LR folder where processed images will be saved.
    :param scale_factor: Factor by which to downscale the images.
    """
    print(f"Starting processing images from {hr_folder_path} to {lr_folder_path} with scale factor {scale_factor}")
    # Create the LR folder if it doesn't exist
    if not os.path.exists(lr_folder_path):
        os.makedirs(lr_folder_path)
        print(f"Created LR folder: {lr_folder_path}")

    # Get the list of image files in the HR folder
    image_files = [f for f in os.listdir(hr_folder_path) if os.path.isfile(os.path.join(hr_folder_path, f))]
    print(f"Found {len(image_files)} images in the HR folder.")

    # Process the images
    for filename in image_files:
        try:
            hr_image_path = os.path.join(hr_folder_path, filename)
            lr_image_filename = filename.replace("HR", "LR")
            lr_image_path = os.path.join(lr_folder_path, lr_image_filename)
            
            # Open the HR image
            with Image.open(hr_image_path) as hr_image:
                print(f"Processing image: {filename}")
                # Process it (downscale)
                lr_image = process_image(hr_image, scale_factor)
                # Save to the LR folder
                lr_image.save(lr_image_path)
                print(f"Processed and saved {filename} as {lr_image_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# # Example usage:
# hr_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/div2k_HR"
# lr_folder = "/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/למידה עמוקה/div2k_sample_dataset/DIV2K_train_LR_bicubic/div2k_LR"
# process_and_save_images(hr_folder, lr_folder, scale_factor=4)
