import pytest
import torch
from src.model.maxsr import MaxSRModel
from src.utils.utils import load_config

@pytest.fixture
def model(config):
    # Initialize the AdaptiveBlockAttention with the test configuration
    return MaxSRModel(config)


@pytest.fixture
def dummy_input():
    # Dummy input tensor for single patch(batch_size, channels, height, width)
    input_patch = torch.randn(1, 3, 64, 64)  # Batch size, Channels, Height, Width
    return input_patch


def test_maxsr_model_output_shape(model, dummy_input, scale_factor=4):
    # Run the dummy input through the AdaptiveBlockAttention
    output = model(dummy_input)
    batch_size, channels, height, width = dummy_input.shape
    assert output.shape == (batch_size, channels, height * scale_factor, width* scale_factor), "Output shape should match input shape."
    # Load configuration
    config = load_config(os.path.join(os.getcwd(), "config", "maxsr_tiny.yaml"))[
        "model_config"
    ]