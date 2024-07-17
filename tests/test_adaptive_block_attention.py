import pytest
import torch

# Adjust the import based on your actual module structure
from src.layers.adaptive_block_attention import AdaptiveBlockAttention

# Configuration for the AdaptiveBlockAttention
config = {
    "self_attention": {
        "in_channels": 64,  # Example: 64 input channels
        "block_size": 16,  # Example: 16x16 blocks
        "num_heads": 8,  # Example: 8 attention heads
    },
    "ffn": {
        "in_channels": 64,  # Should match in_channels of self_attention
        "hidden_dim": 256,  # Example: Larger hidden dimension for FFN
        "dropout": 0.1,  # Example: Dropout rate
    },
}


@pytest.fixture
def model():
    # Initialize the AdaptiveBlockAttention with the test configuration
    return AdaptiveBlockAttention(config)


@pytest.fixture
def dummy_input():
    # Dummy input tensor (batch_size, channels, height, width)
    return torch.rand(2, 64, 64, 64)  # Example: batch size of 2, 64x64 image


def test_output_shape(model, dummy_input):
    # Run the dummy input through the AdaptiveBlockAttention
    output = model(dummy_input)
    # Check the output shape
    assert output.shape == dummy_input.shape, "Output shape should match input shape."


def test_output_values(model, dummy_input):
    # This test could be expanded based on expected output range or properties
    output = model(dummy_input)
    # Check for NaNs or Infs
    assert not torch.isnan(output).any(), "Output should not contain NaNs."
    assert not torch.isinf(output).any(), "Output should not contain Infs."
