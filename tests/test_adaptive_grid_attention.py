import pytest
import torch
from src.layers.adaptive_grid_attention import GridAttention


@pytest.fixture
def setup_grid_attention():
    in_channels = 64
    num_heads = 8
    grid_attention = GridAttention(in_channels, num_heads)
    batch_size = 2
    height = 16
    width = 16
    return grid_attention, in_channels, batch_size, height, width


def test_output_shape(setup_grid_attention):
    grid_attention, in_channels, batch_size, height, width = setup_grid_attention

    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    # Pass the input through the grid attention block
    output = grid_attention(dummy_input)

    # Check if the output shape matches the input shape
    assert output.shape == (batch_size, in_channels, height * width)


def test_output_values(setup_grid_attention):
    grid_attention, in_channels, batch_size, height, width = setup_grid_attention

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    # Pass the input through the grid attention block
    output = grid_attention(dummy_input)

    # Ensure output is not identical to input (indicating some transformation)
    assert not torch.equal(dummy_input.view(batch_size, in_channels, -1), output)
