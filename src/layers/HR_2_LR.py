import torch
import torchvision.transforms as transforms
import PIL.Image as Image

class BicubicDownscaler:
    def __init__(self, scale_factor):
        """
        Initializes the downscaler with the scale factor.
        :param scale_factor: Factor by which to downscale the image.
        """
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()

    def downscale(self, image_path, save_path=None):
        """
        Downscales the image at the given path using bicubic interpolation.
        :param image_path: Path to the input image.
        :param save_path: Optional path to save the downscaled image.
        :return: Downscaled PIL image.
        """
        # Load the image
        image = Image.open(image_path)
        original_size = image.size  # (width, height)

        # Calculate the target size for downscaling
        target_size = (int(original_size[0] / self.scale_factor), int(original_size[1] / self.scale_factor))

        # Define the resize transformation
        resize_transform = transforms.Resize(target_size, interpolation=Image.BICUBIC)

        # Convert to tensor
        image_tensor = self.to_tensor(image).unsqueeze(0)  # Add batch dimension

        # Downscale the image
        downscaled_image_tensor = resize_transform(image_tensor)

        # Convert back to image
        downscaled_image = self.to_image(downscaled_image_tensor.squeeze(0))  # Remove batch dimension

        # Save the image if save_path is provided
        if save_path:
            downscaled_image.save(save_path)

        return downscaled_image

# Example usage:
# Initialize the downscaler with the scale factor
downscaler = BicubicDownscaler(scale_factor=4)  # Example: downscale by a factor of 4

# Downscale an image and save the result
downscaled_image = downscaler.downscale("/Users/matanoz/Downloads/Screenshot 2024-05-22 at 18.41.40.png", "downscaled_image.jpg")

# Optionally, display the downscaled image
downscaled_image.show()
