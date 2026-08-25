"""Validation for photometric augmentation primitives.

Each test enforces a specific property:
- Topological shape contracts (B, C, H, W)
- Value range bounds in [0, 1] after perturbation
- Determinism under a fixed torch.Generator seed
- Guard-clause rejection of invalid parameters
"""

import pytest
import torch

from core.exceptions import TensorTopologyError
from core.math.augmentation import (
    add_gaussian_noise,
    augment_image,
    gaussian_blur,
    jitter_contrast,
)


def _batch() -> torch.Tensor:
    """A deterministic float32 batch in [0, 1]."""
    return torch.linspace(0.0, 1.0, steps=3 * 8 * 8).reshape(1, 3, 8, 8)


def test_add_gaussian_noise_preserves_shape_and_range() -> None:
    batch: torch.Tensor = _batch()

    noisy: torch.Tensor = add_gaussian_noise(batch, sigma=0.03)

    assert noisy.shape == batch.shape
    assert noisy.min().item() >= 0.0
    assert noisy.max().item() <= 1.0


def test_add_gaussian_noise_is_deterministic_under_seed() -> None:
    batch: torch.Tensor = _batch()
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first: torch.Tensor = add_gaussian_noise(batch, sigma=0.03, generator=first_generator)
    second: torch.Tensor = add_gaussian_noise(batch, sigma=0.03, generator=second_generator)

    assert torch.equal(first, second)


def test_add_gaussian_noise_rejects_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        add_gaussian_noise(_batch(), sigma=0.0)
    with pytest.raises(ValueError):
        add_gaussian_noise(_batch(), sigma=0.5)


def test_jitter_contrast_range() -> None:
    jittered: torch.Tensor = jitter_contrast(_batch(), alpha=1.2)

    assert jittered.shape == _batch().shape
    assert jittered.min().item() >= 0.0
    assert jittered.max().item() <= 1.0


def test_jitter_contrast_identity() -> None:
    batch: torch.Tensor = _batch()

    assert torch.allclose(jitter_contrast(batch, alpha=1.0), batch)


def test_gaussian_blur_preserves_shape_and_range() -> None:
    batch: torch.Tensor = _batch()

    blurred: torch.Tensor = gaussian_blur(batch)

    assert blurred.shape == batch.shape
    assert blurred.min().item() >= 0.0
    assert blurred.max().item() <= 1.0


def test_augment_image_composed_pipeline() -> None:
    batch: torch.Tensor = _batch()

    augmented: torch.Tensor = augment_image(batch, noise_sigma=0.02, contrast_alpha=1.1, blur=True)

    assert augmented.shape == batch.shape
    assert augmented.min().item() >= 0.0
    assert augmented.max().item() <= 1.0


def test_augmentation_rejects_non_rank4_tensor() -> None:
    with pytest.raises(TensorTopologyError):
        add_gaussian_noise(torch.randn(3, 8, 8), sigma=0.03)


def test_augmentation_rejects_non_float32() -> None:
    with pytest.raises(TensorTopologyError):
        gaussian_blur(torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8))
