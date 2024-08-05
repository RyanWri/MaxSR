import os
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from IPython.display import display
from utils.utils import next_power_of_two


# Function to get image details from a given directory
def get_image_details(image_folder_path):
    """
    This function extracts the image name, width, and height for each image
    in a specified directory and returns this information as a DataFrame.
    """
    image_details = []

    # Iterate over each file in the directory
    for image_name in os.listdir(image_folder_path):
        if image_name.endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):  # Consider common image formats
            image_path = os.path.join(image_folder_path, image_name)

            # Open the image using PIL
            with Image.open(image_path) as img:
                width, height = img.size
                image_details.append(
                    {"image_name": image_name, "width": width, "height": height}
                )

    # Create a DataFrame from the image details
    df = pd.DataFrame(image_details)

    return df


def analyze_image_details():
    # Paths to the train and validation image directories
    train_image_folder_path = "F:\\לימודים\\תואר שני\\סמסטר ב\\Deel Learning\\Dataset_Preprocess\\DIV2K_train_HR"
    valid_image_folder_path = "F:\\לימודים\\תואר שני\\סמסטר ב\\Deel Learning\\Dataset_Preprocess\\DIV2K_valid_HR"

    # Get image details for train and validation datasets
    train_df = get_image_details(train_image_folder_path)
    valid_df = get_image_details(valid_image_folder_path)

    # Display descriptive statistics of the DataFrames
    print("Train DataFrame:")
    display(train_df.describe())
    print("\nValid DataFrame:")
    display(valid_df.describe())

    # Save the DataFrames to CSV files
    train_df.to_csv("train_image_details.csv", index=False)
    valid_df.to_csv("valid_image_details.csv", index=False)

    # Merge train and valid DataFrames for combined analysis
    combined_df = pd.concat(
        [train_df.assign(dataset="Train"), valid_df.assign(dataset="Valid")]
    )


# Perform Exploratory Data Analysis on Combined Data
def analyze_image_heights(combined_df):
    """
    Analyze the distribution of image heights in the combined dataset,
    print the most common height, and plot the distributions.
    """
    # Count the occurrences of each height
    height_counts = Counter(combined_df["height"])

    # Analyze image heights in combined datasets
    most_common_height = analyze_image_heights(combined_df)

    # Determine target height
    target_height = next_power_of_two(most_common_height)
    print(f"Chosen target height (power of two): {target_height} pixels")

    # Find the most common height
    most_common_height = height_counts.most_common(1)[0][0]

    # Print the most common height
    print(
        f"\nThe most common height in both Train and Validation sets is {most_common_height} pixels."
    )

    # Plot a histogram of the heights
    plt.figure(figsize=(10, 6))
    plt.hist(
        combined_df[combined_df["dataset"] == "Train"]["height"],
        bins=30,
        label="Train Image Heights",
        alpha=0.7,
        edgecolor="k",
    )
    plt.hist(
        combined_df[combined_df["dataset"] == "Valid"]["height"],
        bins=30,
        label="Valid Image Heights",
        alpha=0.7,
        edgecolor="k",
    )
    plt.title("Distribution of Image Heights in Train and Valid Sets")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Comparison of two distributions using KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=combined_df[combined_df["dataset"] == "Train"]["height"],
        label="Train Image Heights",
        fill=True,
    )
    sns.kdeplot(
        data=combined_df[combined_df["dataset"] == "Valid"]["height"],
        label="Valid Image Heights",
        fill=True,
    )
    plt.title("Comparison of Image Height Distributions")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Comparison of two distributions using count plots
    plt.figure(figsize=(14, 7))
    sns.countplot(x="height", data=combined_df, hue="dataset")
    plt.xticks(rotation=90)
    plt.title("Comparison of Image Height Counts Between Train and Valid Sets")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count")
    plt.legend(title="Dataset")
    plt.show()

    # Additional statistics
    print("\nDescriptive Statistics for Heights:")
    print(combined_df["height"].describe())

    return most_common_height


# Function to pad images to a specified size
def pad_image_to_size(image_path, target_width, target_height):
    """
    Pad an image to the target size, keeping the image centered,
    and filling the extra space with black pixels.
    """
    with Image.open(image_path) as img:
        # Calculate padding amounts
        pad_height = target_height - img.height
        pad_width = target_width - img.width

        # Ensure padding is only added if necessary
        if pad_height > 0 or pad_width > 0:
            # Add padding equally on all sides
            padding = (
                pad_width // 2,
                pad_height // 2,
                pad_width - pad_width // 2,
                pad_height - pad_height // 2,
            )
            img_padded = ImageOps.expand(img, padding, fill="black")
        else:
            img_padded = img

        return img_padded


def transform_hr_image_to_pad(
    src_directory: str, target_directory: str, target_width: int, target_height: int
):
    for image_name in os.listdir(src_directory):
        if image_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(src_directory, image_name)
            padded_image = pad_image_to_size(image_path, target_width, target_height)
            padded_image.save(os.path.join(target_directory, image_name))


def downscale_and_adjust(image, scale_factor=0.25, target_mode="RGB"):
    # Downscale using bicubic interpolation
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.BICUBIC)

    # Adjust size to the nearest power of two
    new_width = next_power_of_two(image.width)
    new_height = next_power_of_two(image.height)
    result_image = Image.new(target_mode, (new_width, new_height))
    result_image.paste(
        image, (0, 0)
    )  # Paste resized image into image sized to the next power of two

    return result_image


if __name__ == "__main__":
    train_hr_directory = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR"
    valid_hr_directory = "C:\datasets\DIV2K\Dataset\DIV2K_valid_HR"

    train_output_folder = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD"
    valid_output_folder = "C:\datasets\DIV2K\Dataset\DIV2K_valid_HR_PAD"

    target_width = 2048  # Assuming a fixed width
    target_height = 2048

    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(valid_output_folder, exist_ok=True)
    transform_hr_image_to_pad(
        train_hr_directory, train_output_folder, target_width, target_height
    )
    transform_hr_image_to_pad(
        valid_hr_directory, valid_output_folder, target_width, target_height
    )
