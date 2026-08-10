"""
Loop invariant validation for the composable preprocessing pipeline.

Each test enforces end-to-end pipeline correctness:
- Output shape matches (B, C, target_height, target_width)
- Normalized range [0.0, 1.0] is preserved through the full chain
- Domain type boundary (ImageTensor) is maintained
- Invalid dimensions are rejected at the guard clause
"""
import pytest
import torch

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor
from core.math.preprocessing import preprocess_for_encoder


def test_preprocess_full_pipeline_shape() -> None:
    """Output shape must be (B, C, target_height, target_width)."""
    # Shape: (1, 3, 512, 512) — simulate a high-resolution input
    raw: torch.Tensor = torch.randint(0, 256, (1, 3, 512, 512), dtype=torch.float32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = preprocess_for_encoder(image, target_height=224, target_width=224)

    assert result.raw_tensor.shape == (1, 3, 224, 224)


def test_preprocess_full_pipeline_normalized_range() -> None:
    """All output values must lie within [0.0, 1.0] after normalization + resize."""
    # Shape: (2, 3, 64, 64)
    raw: torch.Tensor = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.float32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = preprocess_for_encoder(image, target_height=32, target_width=32)

    assert result.raw_tensor.min().item() >= 0.0
    assert result.raw_tensor.max().item() <= 1.0


def test_preprocess_returns_image_tensor() -> None:
    """Pipeline output must preserve the ImageTensor domain type."""
    # Shape: (1, 3, 128, 128)
    raw: torch.Tensor = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.float32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = preprocess_for_encoder(image, target_height=64, target_width=64)

    assert isinstance(result, ImageTensor)


def test_preprocess_invalid_dimensions() -> None:
    """Non-positive target dimensions must be rejected at the guard clause."""
    # Shape: (1, 3, 64, 64)
    raw: torch.Tensor = torch.randn(1, 3, 64, 64)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    with pytest.raises(TensorTopologyError):
        preprocess_for_encoder(image, target_height=0, target_width=224)

    with pytest.raises(TensorTopologyError):
        preprocess_for_encoder(image, target_height=224, target_width=-10)
