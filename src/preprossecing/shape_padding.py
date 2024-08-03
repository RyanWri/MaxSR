# %% 
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
import os
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from IPython.display import display
import numpy as np

# Function to get image details from a given directory
def get_image_details(image_folder_path):
    """
    This function extracts the image name, width, and height for each image
    in a specified directory and returns this information as a DataFrame.
    """
    image_details = []
    
    # Iterate over each file in the directory
    for image_name in os.listdir(image_folder_path):
        if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Consider common image formats
            image_path = os.path.join(image_folder_path, image_name)
            
            # Open the image using PIL
            with Image.open(image_path) as img:
                width, height = img.size
                image_details.append({'image_name': image_name, 'width': width, 'height': height})
    
    # Create a DataFrame from the image details
    df = pd.DataFrame(image_details)
    
    return df

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
train_df.to_csv('train_image_details.csv', index=False)
valid_df.to_csv('valid_image_details.csv', index=False)

# Merge train and valid DataFrames for combined analysis
combined_df = pd.concat([train_df.assign(dataset='Train'), valid_df.assign(dataset='Valid')])

# Perform Exploratory Data Analysis on Combined Data
def analyze_image_heights(combined_df):
    """
    Analyze the distribution of image heights in the combined dataset,
    print the most common height, and plot the distributions.
    """
    # Count the occurrences of each height
    height_counts = Counter(combined_df['height'])
    
    # Find the most common height
    most_common_height = height_counts.most_common(1)[0][0]
    
    # Print the most common height
    print(f"\nThe most common height in both Train and Validation sets is {most_common_height} pixels.")
    
    # Plot a histogram of the heights
    plt.figure(figsize=(10, 6))
    plt.hist(combined_df[combined_df['dataset'] == 'Train']['height'], bins=30, label='Train Image Heights', alpha=0.7, edgecolor='k')
    plt.hist(combined_df[combined_df['dataset'] == 'Valid']['height'], bins=30, label='Valid Image Heights', alpha=0.7, edgecolor='k')
    plt.title('Distribution of Image Heights in Train and Valid Sets')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # Comparison of two distributions using KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=combined_df[combined_df['dataset'] == 'Train']['height'], label='Train Image Heights', fill=True)
    sns.kdeplot(data=combined_df[combined_df['dataset'] == 'Valid']['height'], label='Valid Image Heights', fill=True)
    plt.title('Comparison of Image Height Distributions')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Comparison of two distributions using count plots
    plt.figure(figsize=(14, 7))
    sns.countplot(x='height', data=combined_df, hue='dataset')
    plt.xticks(rotation=90)
    plt.title('Comparison of Image Height Counts Between Train and Valid Sets')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    plt.legend(title='Dataset')
    plt.show()

    # Additional statistics
    print("\nDescriptive Statistics for Heights:")
    print(combined_df['height'].describe())

    return most_common_height

# Analyze image heights in combined datasets
most_common_height = analyze_image_heights(combined_df)

# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))

# Determine target height
target_height = next_power_of_two(most_common_height)
print(f"Chosen target height (power of two): {target_height} pixels")

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
            padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
            img_padded = ImageOps.expand(img, padding, fill='black')
        else:
            img_padded = img
        
        return img_padded

# %% 
# Example padding application based on EDA findings
target_width = 2048  # Assuming a fixed width

# Paths to save padded images


train_output_folder = "F:\\לימודים\\תואר שני\\סמסטר ב\\Deel Learning\\Dataset_Preprocess\\DIV2K_train_HR_PAD"
valid_output_folder = "F:\\לימודים\\תואר שני\\סמסטר ב\\Deel Learning\\Dataset_Preprocess\\DIV2K_valid_HR_PAD"
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(valid_output_folder, exist_ok=True)
# %% 
# Apply padding to train images
for image_name in os.listdir(train_image_folder_path):
    if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(train_image_folder_path, image_name)
        padded_image = pad_image_to_size(image_path, target_width, target_height)
        padded_image.save(os.path.join(train_output_folder, image_name))
# %% 
# Apply padding to validation images
for image_name in os.listdir(valid_image_folder_path):
    if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(valid_image_folder_path, image_name)
        padded_image = pad_image_to_size(image_path, target_width, target_height)
        padded_image.save(os.path.join(valid_output_folder, image_name))

# Explanation of padding
print("\nPadding Explanation:")
print("Padding involves adding pixels to the borders of an image to achieve a desired size.")
print("This is done to ensure that all images have consistent dimensions for model training.")
print("Padding can help preserve features and avoid downsampling issues in convolutional layers.")
