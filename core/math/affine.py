import torch
import torch.nn.functional as F

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor


def apply_affine_transformation(image: ImageTensor, theta: torch.Tensor) -> ImageTensor:
    """
    Applies an affine transformation to the image tensor.

    Args:
        image (ImageTensor): Image tensor of shape (B, C, H, W).
        theta (torch.Tensor): Affine transformation matrix. Shape: (B, 2, 3).

    Returns:
        ImageTensor: The transformed tensor, shape (B, C, H, W).

    Raises:
        TensorTopologyError: If theta has the wrong shape or batch size.
    """
    if not isinstance(theta, torch.Tensor):
        raise TensorTopologyError("Transformation matrix must be a torch.Tensor.")

    if theta.ndim != 3 or theta.shape[1:] != (2, 3):
        raise TensorTopologyError(
            f"Invalid affine matrix topology. Expected shape (B, 2, 3), got {tuple(theta.shape)}."
        )

    if theta.shape[0] != image.raw_tensor.shape[0]:
        raise TensorTopologyError(
            f"Batch size mismatch. Image batch: {image.raw_tensor.shape[0]}, "
            f"Theta batch: {theta.shape[0]}."
        )

    # Shape: (B, C, H, W)
    batch_size, channels, height, width = image.raw_tensor.shape

    # Build the sampling grid.
    grid = F.affine_grid(theta, size=[batch_size, channels, height, width], align_corners=False)

    # Sample the image at the grid points.
    transformed_raw = F.grid_sample(
        image.raw_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return ImageTensor(raw_tensor=transformed_raw)
