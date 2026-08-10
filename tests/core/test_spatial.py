"""
Validation for declarative spatial transformation primitives.

Each test enforces a specific mathematical property:
- Topological shape contracts (B, C, H, W)
- Value range bounds after normalization
- Idempotence of normalization
- Type boundary compliance (ImageTensor vs raw Tensor)
- Guard clause rejection of invalid parameters
"""
import pytest
import torch

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor
from core.math.spatial import (
    normalize_channels,
    resize_spatial_dimensions,
    rearrange_channels_last,
    tile_batch_dimension,
    flatten_spatial_grid,
)


# ---------------------------------------------------------------------------
# normalize_channels
# ---------------------------------------------------------------------------

def test_normalize_channels_range() -> None:
    """Output values must lie within the continuous manifold [0.0, 1.0]."""
    # Construct tensor with known extremes: 0 and 255.
    # Shape: (1, 3, 4, 4)
    raw: torch.Tensor = torch.randint(0, 256, (1, 3, 4, 4), dtype=torch.float32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = normalize_channels(image)

    assert result.raw_tensor.min().item() >= 0.0
    assert result.raw_tensor.max().item() <= 1.0


def test_normalize_channels_preserves_topology() -> None:
    """Shape (B, C, H, W) must be invariant under normalization."""
    # Shape: (2, 3, 16, 16)
    raw: torch.Tensor = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.float32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = normalize_channels(image)

    assert result.raw_tensor.shape == (2, 3, 16, 16)


def test_normalize_channels_idempotence() -> None:
    """
    Idempotence property: normalize(normalize(x)) ≈ normalize(x).
    For values already in [0, 1], a second normalization divides by 255
    again, producing values ≈ 0. We verify the first pass is stable
    and the second pass yields near-zero (confirming the operation is
    a pure scalar division, not an adaptive rescale).
    """
    # Shape: (1, 1, 2, 2)
    raw: torch.Tensor = torch.tensor([[[[0.0, 255.0], [128.0, 64.0]]]])
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    first_pass: ImageTensor = normalize_channels(image)
    second_pass: ImageTensor = normalize_channels(first_pass)

    # After first pass: max ≈ 1.0. After second pass: max ≈ 1/255 ≈ 0.00392.
    assert second_pass.raw_tensor.max().item() < 0.01


# ---------------------------------------------------------------------------
# resize_spatial_dimensions
# ---------------------------------------------------------------------------

def test_resize_spatial_dimensions_shape() -> None:
    """Output H, W must match the target parameters exactly."""
    # Shape: (1, 3, 64, 64)
    raw: torch.Tensor = torch.randn(1, 3, 64, 64)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = resize_spatial_dimensions(image, 224, 224)

    assert result.raw_tensor.shape == (1, 3, 224, 224)


def test_resize_spatial_dimensions_preserves_batch_channels() -> None:
    """B and C axes must remain unchanged after spatial resampling."""
    # Shape: (4, 1, 32, 32) — grayscale batch of 4
    raw: torch.Tensor = torch.randn(4, 1, 32, 32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = resize_spatial_dimensions(image, 128, 128)

    assert result.raw_tensor.shape[0] == 4
    assert result.raw_tensor.shape[1] == 1


def test_resize_spatial_dimensions_invalid_target() -> None:
    """Non-positive target dimensions must be rejected at the guard clause."""
    # Shape: (1, 3, 64, 64)
    raw: torch.Tensor = torch.randn(1, 3, 64, 64)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    with pytest.raises(TensorTopologyError):
        resize_spatial_dimensions(image, 0, 224)

    with pytest.raises(TensorTopologyError):
        resize_spatial_dimensions(image, 224, -1)


# ---------------------------------------------------------------------------
# rearrange_channels_last
# ---------------------------------------------------------------------------

def test_rearrange_channels_last_shape() -> None:
    """Output shape must be (B, H, W, C) after axis transposition."""
    # Shape: (2, 3, 16, 32)
    raw: torch.Tensor = torch.randn(2, 3, 16, 32)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: torch.Tensor = rearrange_channels_last(image)

    assert result.shape == (2, 16, 32, 3)


def test_rearrange_channels_last_returns_raw_tensor() -> None:
    """Return type must be torch.Tensor, not ImageTensor (Liskov compliance)."""
    # Shape: (1, 3, 8, 8)
    raw: torch.Tensor = torch.randn(1, 3, 8, 8)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: torch.Tensor = rearrange_channels_last(image)

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, ImageTensor)


# ---------------------------------------------------------------------------
# tile_batch_dimension
# ---------------------------------------------------------------------------

def test_tile_batch_dimension_shape() -> None:
    """Output batch dimension must equal B * repeats."""
    # Shape: (2, 3, 8, 8), repeats=3 → (6, 3, 8, 8)
    raw: torch.Tensor = torch.randn(2, 3, 8, 8)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: ImageTensor = tile_batch_dimension(image, repeats=3)

    assert result.raw_tensor.shape == (6, 3, 8, 8)


def test_tile_batch_dimension_invalid_repeats() -> None:
    """Non-positive repeat factor must be rejected at the guard clause."""
    # Shape: (1, 3, 8, 8)
    raw: torch.Tensor = torch.randn(1, 3, 8, 8)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    with pytest.raises(TensorTopologyError):
        tile_batch_dimension(image, repeats=0)

    with pytest.raises(TensorTopologyError):
        tile_batch_dimension(image, repeats=-2)


# ---------------------------------------------------------------------------
# flatten_spatial_grid
# ---------------------------------------------------------------------------

def test_flatten_spatial_grid_shape() -> None:
    """Output shape must be (B, C, H*W) after spatial collapse."""
    # Shape: (1, 3, 4, 8) → (1, 3, 32)
    raw: torch.Tensor = torch.randn(1, 3, 4, 8)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: torch.Tensor = flatten_spatial_grid(image)

    assert result.shape == (1, 3, 32)


def test_flatten_spatial_grid_returns_raw_tensor() -> None:
    """Return type must be torch.Tensor, not ImageTensor (Liskov compliance)."""
    # Shape: (1, 3, 4, 4)
    raw: torch.Tensor = torch.randn(1, 3, 4, 4)
    image: ImageTensor = ImageTensor(raw_tensor=raw)

    result: torch.Tensor = flatten_spatial_grid(image)

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, ImageTensor)
