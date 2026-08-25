"""
Preprocessing pipeline for the vision encoder.

Chains the spatial primitives from core.math.spatial into a single call.
Pure function: no I/O, no side effects, no global state.

Reference: Goodfellow et al., Deep Learning, Ch. 8 — input normalization
for stable training of neural networks.
"""

from core.exceptions import TensorTopologyError
from core.math.spatial import normalize_channels, resize_spatial_dimensions
from core.models.value_objects import ImageTensor


def preprocess_for_encoder(
    image: ImageTensor,
    target_height: int,
    target_width: int,
) -> ImageTensor:
    """
    Prepares an image for the vision encoder.

    Pipeline:
        1. normalize_channels: [0, 255] → [0.0, 1.0]
        2. resize_spatial_dimensions: (H, W) → (target_height, target_width)

    The result keeps the (B, C, H, W) shape required by ImageTensor.
    Per-channel ImageNet standardization (μ, σ) is deferred to Phase 3,
    once the encoder architecture is chosen.

    Args:
        image: Image tensor of shape (B, C, H, W).
        target_height: Output height. Must be > 0.
        target_width: Output width. Must be > 0.

    Returns:
        ImageTensor of shape (B, C, target_height, target_width),
        values in [0.0, 1.0].

    Raises:
        TensorTopologyError: If target dimensions are non-positive.

    Temporal complexity: O(N) where N is the number of pixels (vectorized operations).
    """
    if target_height <= 0 or target_width <= 0:
        raise TensorTopologyError(
            f"Target spatial dimensions must be positive. "
            f"Got height={target_height}, width={target_width}."
        )

    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Step 1: normalize pixel values. Shape stays (B, C, H, W).
    normalized: ImageTensor = normalize_channels(image)

    # Step 2: resize. Shape: (B, C, H, W) → (B, C, tH, tW).
    resized: ImageTensor = resize_spatial_dimensions(normalized, target_height, target_width)

    return resized
