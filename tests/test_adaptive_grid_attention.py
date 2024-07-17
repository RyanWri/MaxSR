# File: tests/test_adaptive_grid_attention.py

import pytest
import torch
from src.layers.adaptive_grid_attention import AdaptiveGridAttention


@pytest.fixture
def config():
    return {
        "grid_attention": {"in_channels": 64, "block_size": 16, "num_heads": 8},
        "ffn": {"in_channels": 64, "hidden_dim": 256, "dropout": 0.1},
    }


@pytest.fixture
def adaptive_grid_attention(config):
    return AdaptiveGridAttention(config)


def test_adaptive_grid_attention_shape(adaptive_grid_attention):
    dummy_input = torch.rand(2, 64, 64, 64)
    output = adaptive_grid_attention(dummy_input)
    assert output.shape == dummy_input.shape, "Output shape should match input shape"
