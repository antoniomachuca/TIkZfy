"""
Spatial transformation primitives using einops.

Each function is a pure, side-effect-free operation on tensors. Functions
whose output does not have the (B, C, H, W) shape required by ImageTensor
return a raw torch.Tensor instead.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor


def normalize_channels(image: ImageTensor) -> ImageTensor:
    """
    Normalizes pixel values from [0, 255] to [0.0, 1.0].

    Args:
        image: Image tensor of shape (B, C, H, W).

    Returns:
        ImageTensor with the same shape, float32 dtype, values in [0, 1].

    Temporal complexity: O(N) where N is tensor size (vectorized division).
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # Shape: (B, C, H, W) — element-wise division.
    normalized: torch.Tensor = image.raw_tensor.to(dtype=torch.float32) / 255.0

    return ImageTensor(raw_tensor=normalized)


def resize_spatial_dimensions(
    image: ImageTensor,
    target_height: int,
    target_width: int,
) -> ImageTensor:
    """
    Resizes spatial dimensions (H, W) via bilinear interpolation.

    Args:
        image: Image tensor of shape (B, C, H, W).
        target_height: Output height. Must be > 0.
        target_width: Output width. Must be > 0.

    Returns:
        ImageTensor of shape (B, C, target_height, target_width).

    Raises:
        TensorTopologyError: If target dimensions are non-positive.

    Temporal complexity: O(N) (vectorized interpolation).
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
    Transposes axes: (B, C, H, W) → (B, H, W, C).

    Returns a raw torch.Tensor because the output does not have the (B, C, H, W)
    shape required by ImageTensor.

    Args:
        image: Image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor of shape (B, H, W, C).

    Temporal complexity: O(1) (stride permutation, view; no copy).
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # (B, C, H, W) → (B, H, W, C)
    result: torch.Tensor = rearrange(image.raw_tensor, "b c h w -> b h w c")

    return result


def tile_batch_dimension(image: ImageTensor, repeats: int) -> ImageTensor:
    """
    Repeats the batch axis: (B, C, H, W) → (B*repeats, C, H, W).

    Args:
        image: Image tensor of shape (B, C, H, W).
        repeats: Batch replication factor. Must be > 0.

    Returns:
        ImageTensor of shape (B * repeats, C, H, W).

    Raises:
        TensorTopologyError: If repeats is non-positive.

    Temporal complexity: O(N) where N is the output size (memory allocation).
    """
    if repeats <= 0:
        raise TensorTopologyError(f"Repeat factor must be positive. Got repeats={repeats}.")

    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # (B, C, H, W) → (B*repeats, C, H, W)
    tiled: torch.Tensor = repeat(image.raw_tensor, "b c h w -> (b repeat) c h w", repeat=repeats)

    return ImageTensor(raw_tensor=tiled)


def flatten_spatial_grid(image: ImageTensor) -> torch.Tensor:
    """
    Flattens spatial dimensions: (B, C, H, W) → (B, C, H*W).

    Returns a raw torch.Tensor because the output does not have the 4D
    (B, C, H, W) shape required by ImageTensor.

    Args:
        image: Image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor of shape (B, C, H * W).

    Temporal complexity: O(1) if the tensor is contiguous, else O(N).
    """
    if not isinstance(image, ImageTensor):
        raise TypeError("Input must be an ImageTensor instance.")

    # (B, C, H, W) → (B, C, H*W)
    result: torch.Tensor = rearrange(image.raw_tensor, "b c h w -> b c (h w)")

    return result
