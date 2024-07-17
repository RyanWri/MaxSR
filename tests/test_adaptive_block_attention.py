import pytest
import torch

# Adjust the import based on your actual module structure
from src.layers.adaptive_block_attention import AdaptiveBlockAttention

config = {
    "in_channels": 48,
    "out_channels": 48,
    "num_heads": 4,
    "hidden_dim": 128,
    "kernel_size": 3,
    "ffn_in_channels": 48,
    "ffn_out_channels": 48,
}


@pytest.fixture
def model():
    # Initialize the AdaptiveBlockAttention with the test configuration
    return AdaptiveBlockAttention(config)


@pytest.fixture
def dummy_input():
    # Dummy input tensor (batch_size, channels, height, width)
    return torch.rand(1, 48, 256, 256)


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
