import torch.nn as nn


# might need to add config and channels later
class SFEB(nn.Module):
    """
    Shallow Feature Extraction Block (SFEB) for extracting low-level features from the input image.
    The SFEB consists of two convolutional layers without activation functions.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SFEB, self).__init__()
        # First convolutional layer: 1 input channel (grayscale image), 64 output channels, 3x3 kernel, padding of 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # Second convolutional layer: 64 input channels (output from first layer), 64 output channels, 3x3 kernel, padding of 1
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        Forward pass for the SFEB.

        Args:
        x (torch.Tensor): Input image tensor.

        Returns:
        torch.Tensor: Feature map after the first convolutional layer (F_minus_1).
        torch.Tensor: Feature map after the second convolutional layer (F0).
        """
        F_minus_1 = self.conv1(x)
        F0 = self.conv2(F_minus_1)
        return F_minus_1, F0
