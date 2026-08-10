"""
Declarative spatial transformation primitives using einops.

Each function is a single-responsibility, side-effect-free mathematical
operation on tensors. Functions whose output violates the (B, C, H, W)
invariant return raw torch.Tensor to preserve Liskov compliance of
ImageTensor.

Reference: Golub & Van Loan, Matrix Computations — vectorized axis
permutation via stride manipulation in O(1) logical time.
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor


def normalize_channels(image: ImageTensor) -> ImageTensor:
    """
    Maps the integer pixel lattice [0, 255] to the continuous float
    manifold [0.0, 1.0] via scalar division over the full tensor.

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).

    Returns:
        ImageTensor with identical shape, dtype float32, values in [0, 1].

    Temporal complexity: O(1) logical — single vectorized division.
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Shape: (B, C, H, W) — vectorized element-wise division.
    normalized: torch.Tensor = image.raw_tensor.to(dtype=torch.float32) / 255.0

    return ImageTensor(raw_tensor=normalized)


def resize_spatial_dimensions(
    image: ImageTensor,
    target_height: int,
    target_width: int,
) -> ImageTensor:
    """
    Resamples spatial axes (H, W) via bilinear interpolation without
    sequential pixel iteration.

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).
        target_height: Desired output height. Must be > 0.
        target_width: Desired output width. Must be > 0.

    Returns:
        ImageTensor with shape (B, C, target_height, target_width).

    Raises:
        TensorTopologyError: If target dimensions are non-positive.

    Temporal complexity: O(1) logical — single vectorized interpolation.
    """
    if target_height <= 0 or target_width <= 0:
        raise TensorTopologyError(
            f"Target spatial dimensions must be positive. "
            f"Got height={target_height}, width={target_width}."
        )

    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Shape: (B, C, H, W) → (B, C, target_height, target_width)
    resized: torch.Tensor = F.interpolate(
        image.raw_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    return ImageTensor(raw_tensor=resized)


def rearrange_channels_last(image: ImageTensor) -> torch.Tensor:
    """
    Declarative axis transposition: (B, C, H, W) → (B, H, W, C).

    Returns raw torch.Tensor because the output violates the (B, C, H, W)
    structural invariant of ImageTensor.

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).

    Returns:
        torch.Tensor with shape (B, H, W, C).

    Temporal complexity: O(1) logical — stride permutation, zero copy.
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Einops declarative rearrangement: (B, C, H, W) → (B, H, W, C)
    result: torch.Tensor = rearrange(image.raw_tensor, "b c h w -> b h w c")

    return result


def tile_batch_dimension(image: ImageTensor, repeats: int) -> ImageTensor:
    """
    Replicates the batch axis N times: (B, C, H, W) → (B*N, C, H, W).

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).
        repeats: Number of times to replicate along the batch axis. Must be > 0.

    Returns:
        ImageTensor with shape (B * repeats, C, H, W).

    Raises:
        TensorTopologyError: If repeats is non-positive.

    Temporal complexity: O(1) logical — einops repeat with stride expansion.
    """
    if repeats <= 0:
        raise TensorTopologyError(
            f"Repeat factor must be positive. Got repeats={repeats}."
        )

    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Einops declarative repeat: (B, C, H, W) → (B*repeats, C, H, W)
    tiled: torch.Tensor = repeat(
        image.raw_tensor, "b c h w -> (b repeat) c h w", repeat=repeats
    )

    return ImageTensor(raw_tensor=tiled)


def flatten_spatial_grid(image: ImageTensor) -> torch.Tensor:
    """
    Collapses spatial dimensions into a single sequence axis:
    (B, C, H, W) → (B, C, H*W).

    Returns raw torch.Tensor because the output violates the 4D (B, C, H, W)
    structural invariant of ImageTensor.

    Args:
        image: Domain tensor constrained to shape (B, C, H, W).

    Returns:
        torch.Tensor with shape (B, C, H * W).

    Temporal complexity: O(1) logical — contiguous view reshape.
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Einops declarative flatten: (B, C, H, W) → (B, C, H*W)
    result: torch.Tensor = rearrange(image.raw_tensor, "b c h w -> b c (h w)")

    return result
