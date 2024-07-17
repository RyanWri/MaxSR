import pytest
import torch

# Adjust the import based on your actual module structure
from src.layers.adaptive_block_attention import AdaptiveBlockAttention

config = {
    "in_channels": 24,
    "out_channels": 24,
    "num_heads": 2,
    "hidden_dim": 64,
    "kernel_size": 3,
    "ffn_in_channels": 24,
    "ffn_out_channels": 24,
}


@pytest.fixture
def model():
    # Initialize the AdaptiveBlockAttention with the test configuration
    return AdaptiveBlockAttention(config)


@pytest.fixture
def dummy_input():
    # Dummy input tensor (batch_size, channels, height, width)
    return torch.rand(1, 24, 64, 64)


def test_output_shape_and_no_nans(model, dummy_input):
    # Run the dummy input through the AdaptiveBlockAttention
    output = model(dummy_input)
    s = output.shape
    # Check the output shape
    assert output.shape == dummy_input.shape, "Output shape should match input shape."
    assert not torch.isnan(output).any(), "Output should not contain NaNs."
    assert not torch.isinf(output).any(), "Output should not contain Infs."
