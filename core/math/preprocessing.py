"""
Composable preprocessing pipeline for vision encoder consumption.

Chains the spatial primitives from core.math.spatial into a canonical
preprocessing sequence. Pure function: zero I/O, zero side effects,
zero global state mutation.

Reference: Goodfellow et al., Deep Learning, Ch. 8 — input normalization
for stable gradient propagation through the encoder layers.
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
    Canonical preprocessing composition for the vision encoder.

    Pipeline:
        1. normalize_channels: [0, 255] → [0.0, 1.0]
        2. resize_spatial_dimensions: (H, W) → (target_height, target_width)

    The output preserves the (B, C, H, W) invariant and is ready for
    direct encoder consumption. Per-channel ImageNet standardization
    (μ, σ) is deferred to Phase 3 model instantiation (27 Jul) when
    the specific encoder architecture is selected.

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).
        target_height: Encoder-expected spatial height. Must be > 0.
        target_width: Encoder-expected spatial width. Must be > 0.

    Returns:
        ImageTensor with shape (B, C, target_height, target_width),
        values in [0.0, 1.0].

    Raises:
        TensorTopologyError: If target dimensions are non-positive.

    Temporal complexity: O(1) logical — two vectorized passes.
    """
    if target_height <= 0 or target_width <= 0:
        raise TensorTopologyError(
            f"Target spatial dimensions must be positive. "
            f"Got height={target_height}, width={target_width}."
        )

    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Step 1: Pixel lattice normalization. Shape preserved: (B, C, H, W).
    normalized: ImageTensor = normalize_channels(image)

    # Step 2: Spatial resampling. Shape: (B, C, H, W) → (B, C, tH, tW).
    resized: ImageTensor = resize_spatial_dimensions(
        normalized, target_height, target_width
    )

    return resized
