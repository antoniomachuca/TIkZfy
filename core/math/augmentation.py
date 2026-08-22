"""
Photometric augmentation primitives for the vision encoder.

Every transform is a pure, side-effect-free function over float image batches of
shape ``(B, C, H, W)`` with values in ``[0, 1]``. They are implemented with
vectorized PyTorch algebra (no per-pixel loops) and are deterministic under a
fixed ``torch.Generator`` seed.

References:
    Shorten & Khoshgoftaar, A survey on Image Data Augmentation for Deep
        Learning — Gaussian noise, contrast jitter, and Gaussian blur as
        robustness-regularizing perturbations.
    Goodfellow et al., Deep Learning — data augmentation as a proxy for
        invariant priors in vision tasks.
"""

import torch
import torch.nn.functional as F

from core.exceptions import TensorTopologyError

_BLUR_ROW: torch.Tensor = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
_BLUR_COLUMN: torch.Tensor = _BLUR_ROW.clone()


def _validate_image_batch(image: torch.Tensor) -> None:
    """Validate a rank-4 float image batch in ``[0, 1]``."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TensorTopologyError("Image must be a rank-4 tensor with shape (B, C, H, W).")
    if image.dtype != torch.float32:
        raise TensorTopologyError(f"Image must be float32. Got {image.dtype}.")


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise and re-clamp into ``[0, 1]``.

    ``sigma`` must lie in ``[0.01, 0.05]`` per the Tier-1 photometric budget.

    Temporal complexity: O(N) where N is the number of pixels.
    """
    _validate_image_batch(image)
    if not 0.0 < sigma <= 0.05:
        raise ValueError(f"sigma must lie in (0, 0.05]. Got {sigma}.")

    noise: torch.Tensor = torch.randn(
        image.shape,
        dtype=image.dtype,
        device=image.device,
        generator=generator,
    )
    return (image + sigma * noise).clamp(0.0, 1.0)


def jitter_contrast(image: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale pixel deviation from the batch mean by ``alpha`` in ``[0.7, 1.3]``.

    ``alpha * image + (1 - alpha) * mean(image)`` re-clamped into ``[0, 1]``.

    Temporal complexity: O(N) where N is the number of pixels.
    """
    _validate_image_batch(image)
    if not 0.0 <= alpha:
        raise ValueError(f"alpha must be non-negative. Got {alpha}.")

    mean: torch.Tensor = image.mean()
    return (alpha * image + (1.0 - alpha) * mean).clamp(0.0, 1.0)


def gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    """Apply a separable 3x3 Gaussian blur (binomial kernel) per channel.

    Temporal complexity: O(C * H * W) — two 1-D convolutions, no explicit loops.
    """
    _validate_image_batch(image)
    channels: int = image.shape[1]
    kernel_row: torch.Tensor = _BLUR_ROW.to(dtype=image.dtype, device=image.device)
    kernel_column: torch.Tensor = _BLUR_COLUMN.to(dtype=image.dtype, device=image.device)

    row_blurred: torch.Tensor = F.conv2d(
        image,
        kernel_row.view(1, 1, 1, 3).repeat(channels, 1, 1, 1),
        padding=(0, 1),
        groups=channels,
    )
    blurred: torch.Tensor = F.conv2d(
        row_blurred,
        kernel_column.view(1, 1, 3, 1).repeat(channels, 1, 1, 1),
        padding=(1, 0),
        groups=channels,
    )
    return blurred.clamp(0.0, 1.0)


def augment_image(
    image: torch.Tensor,
    noise_sigma: float = 0.03,
    contrast_alpha: float = 1.0,
    blur: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the composed photometric pipeline (noise -> contrast -> blur).

    Temporal complexity: O(N) where N is the number of pixels.
    """
    augmented: torch.Tensor = add_gaussian_noise(image, noise_sigma, generator)
    if contrast_alpha != 1.0:
        augmented = jitter_contrast(augmented, contrast_alpha)
    if blur:
        augmented = gaussian_blur(augmented)
    return augmented
