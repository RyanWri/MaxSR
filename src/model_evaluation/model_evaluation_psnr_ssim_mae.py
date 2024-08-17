import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def downscale_image(image, scale):
    """Downscale the image by a given scale factor."""
    lr_image = resize(
        image, (image.shape[0] // scale, image.shape[1] // scale), anti_aliasing=True
    )
    return lr_image


def upscale_image(lr_image, target_shape):
    """Upscale the image to the target shape using a dummy upscale function."""
    hr_image = resize(lr_image, target_shape, anti_aliasing=True)
    return hr_image


def calculate_metrics(original, generated):
    """Calculate PSNR, SSIM, and MAE between original and generated images."""
    data_range = original.max() - original.min()
    win_size = min(original.shape[0], original.shape[1], 7)  # Ensure win_size is valid
    ssim_value = ssim(
        original, generated, data_range=data_range, win_size=win_size, channel_axis=-1
    )
    psnr_value = psnr(original, generated, data_range=data_range)
    mae_value = np.mean(np.abs(original - generated))
    return psnr_value, ssim_value, mae_value


def simulate_evaluation(hr_images, scales):
    """Simulate model performance evaluation using HR and LR images for different scales."""
    results = []
    comparison_images = {
        i: {scale: [] for scale in scales} for i in range(len(hr_images))
    }

    for i, hr_image in enumerate(hr_images):
        for scale in scales:
            hr_img_name = f"Image {i} x{scale}"

            # Downscale to create a simulated LR image
            lr_image = downscale_image(hr_image, scale)

            # Upscale to simulate the super-resolution model output
            upscaled_image = upscale_image(lr_image, hr_image.shape)

            psnr_value, ssim_value, mae_value = calculate_metrics(
                hr_image, upscaled_image
            )
            results.append(
                {
                    "Scale": f"x{scale}",
                    "Image": f"Image {i}",
                    "PSNR (dB)": psnr_value,
                    "SSIM": ssim_value,
                    "MAE": mae_value,
                }
            )

            comparison_images[i][scale] = (hr_image, lr_image, upscaled_image)

    results_df = pd.DataFrame(results)
    return results_df, comparison_images


def display_comparison(comparison_images):
    """Display comparison between original HR, LR, and upscaled images for each image and each scale."""
    for i, scales_images in comparison_images.items():
        fig, axes = plt.subplots(
            len(scales_images), 3, figsize=(15, 5 * len(scales_images))
        )

        for j, (scale, images) in enumerate(scales_images.items()):
            hr_img, lr_img, upscaled_img = images

            axes[j][0].imshow(hr_img)
            axes[j][0].set_title(f"Original HR - Image {i} x{scale}")
            axes[j][0].axis("off")

            axes[j][1].imshow(lr_img)
            axes[j][1].set_title(f"Low Resolution x{scale}")
            axes[j][1].axis("off")

            axes[j][2].imshow(upscaled_img)
            axes[j][2].set_title(f"Upscaled x{scale}")
            axes[j][2].axis("off")

        plt.show()


def highlight_max_min(data):
    """Highlight the maximum values for PSNR and SSIM, and the minimum values for MAE in a DataFrame."""
    max_attr = "color: green; font-weight: bold"
    min_attr = "color: red; font-weight: bold"
    if data.ndim == 1:  # Series
        is_max = data == data.max()
        is_min = data == data.min()
        return [
            max_attr if v else min_attr if u else "" for v, u in zip(is_max, is_min)
        ]
    else:  # DataFrame
        is_max = data == data.max(axis=0)
        is_min = data == data.min(axis=0)
        return pd.DataFrame(
            np.where(is_max, max_attr, np.where(is_min, min_attr, "")),
            index=data.index,
            columns=data.columns,
        )


def display_results(results_df):
    """Display the evaluation results using a pandas DataFrame with highlighted max and min values."""
    results_df["PSNR (dB)"] = results_df["PSNR (dB)"]
    results_df["SSIM"] = results_df["SSIM"]
    results_df["MAE"] = results_df["MAE"]

    # Compute the averages
    averages = (
        results_df.groupby("Scale")[["PSNR (dB)", "SSIM", "MAE"]].mean().reset_index()
    )
    averages["Image"] = "Average"

    # Sort by Scale and Image to maintain the structure
    results_df = results_df.sort_values(by=["Scale", "Image"])

    # Append the average rows to the results_df
    results_df = pd.concat([results_df, averages], ignore_index=True)

    # Prepare the table for display
    results_df = results_df[["Scale", "Image", "PSNR (dB)", "SSIM", "MAE"]]
    results_df = results_df.set_index(["Scale", "Image"])

    # Apply highlighting function to each group
    def apply_highlight(group):
        return group.style.apply(
            highlight_max_min, subset=["PSNR (dB)", "SSIM", "MAE"], axis=0
        )

    styled_results = results_df.groupby(level=0).apply(apply_highlight)

    # Combine the styled DataFrames
    combined_html = "\n".join(df.to_html() for df in styled_results)
    display(HTML(combined_html))
