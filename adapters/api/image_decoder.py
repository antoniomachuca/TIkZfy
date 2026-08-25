"""Infrastructural image decoding and normalization for HTTP payloads."""

import io
from typing import Any

import numpy as np
import torch
import torchvision.io as tio
from PIL import Image

from core.exceptions import TensorTopologyError
from core.math.spatial import resize_spatial_dimensions
from core.models.value_objects import ImageTensor
from ports.outbound import ImageLoaderPort


def decode_image_bytes_to_tensor(
    image_bytes: bytes,
    target_height: int = 224,
    target_width: int = 224,
) -> ImageTensor:
    """Decode binary image bytes into a normalized ImageTensor with shape (1, 3, H, W).

    Args:
        image_bytes (bytes): Binary payload of the uploaded image (PNG, JPEG, WebP).
        target_height (int): Target height spatial dimension. Default is 224.
        target_width (int): Target width spatial dimension. Default is 224.

    Returns:
        ImageTensor: Normalized float32 tensor of shape (1, 3, target_height, target_width)
                     with intensities in [0.0, 1.0].

    Raises:
        ValueError: If image bytes are empty, target dimensions are non-positive,
                    or decoding fails.
        TensorTopologyError: If decoded tensor has unsupported channel dimensions.

    Temporal complexity: O(H * W) via vectorized tensor normalization and interpolation.
    """
    if not image_bytes:
        raise ValueError("Image bytes payload must be non-empty.")
    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"Target spatial dimensions must be positive. Got ({target_height}, {target_width})."
        )

    decoded: torch.Tensor
    try:
        raw_buffer: torch.Tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        decoded = tio.decode_image(raw_buffer).to(dtype=torch.float32)
        if bool((decoded.max() > 1.0).item()):
            decoded = decoded / 255.0
    except Exception:
        try:
            pil_image: Image.Image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_array: np.ndarray[Any, np.dtype[np.float32]] = (
                np.array(pil_image, dtype=np.float32) / 255.0
            )
            # Shape: (3, H, W)
            decoded = torch.from_numpy(np_array).permute(2, 0, 1)
        except Exception as exc:
            raise ValueError(f"Failed to decode image bytes: {exc}") from exc

    # Channel normalization to 3 channels (RGB)
    channels: int = int(decoded.shape[0])
    normalized_channels: torch.Tensor
    if channels == 1:
        # Grayscale -> RGB: repeat across channel dimension
        # Shape: (3, H, W)
        normalized_channels = decoded.repeat(3, 1, 1)
    elif channels == 4:
        # RGBA -> RGB: slice out alpha channel
        # Shape: (3, H, W)
        normalized_channels = decoded[:3, :, :]
    elif channels == 3:
        # Standard RGB
        # Shape: (3, H, W)
        normalized_channels = decoded
    else:
        raise TensorTopologyError(f"Unsupported image channel count: {channels}.")

    # Shape: (1, 3, H, W)
    batched_tensor: ImageTensor = ImageTensor(raw_tensor=normalized_channels.unsqueeze(0))

    return resize_spatial_dimensions(
        image=batched_tensor,
        target_height=target_height,
        target_width=target_width,
    )


class ByteImageLoader(ImageLoaderPort):
    """Infrastructural adapter implementing ImageLoaderPort for binary image bytes."""

    def load_image(self, source_path: str) -> ImageTensor:
        """Load an image from disk and decode it to an ImageTensor.

        Args:
            source_path (str): Absolute file system path to the image file.

        Returns:
            ImageTensor: Normalized tensor with shape (1, 3, 224, 224).
        """
        with open(source_path, "rb") as file_handle:
            payload: bytes = file_handle.read()
        return decode_image_bytes_to_tensor(payload)


__all__ = ["ByteImageLoader", "decode_image_bytes_to_tensor"]
