import torch
import torch.nn.functional as F

from core.exceptions import TensorTopologyError
from core.models.value_objects import ImageTensor


def apply_affine_transformation(image: ImageTensor, theta: torch.Tensor) -> ImageTensor:
    """
    Applies an affine transformation to the given image tensor without sequential
    spatial iteration, ensuring O(1) logical time execution.

    Args:
        image (ImageTensor): The domain tensor constrained to shape (B, C, H, W).
        theta (torch.Tensor): The affine transformation matrix.
                              Shape: (B, 2, 3).

    Returns:
        ImageTensor: The transformed tensor with identical topological constraints.
                     Shape: (B, C, H, W).

    Raises:
        TensorTopologyError: If the affine matrix dimensionality is violated.
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

    # O(1) vectorized grid generation.
    grid = F.affine_grid(theta, size=[batch_size, channels, height, width], align_corners=False)

    # Parallel bilinear sampling.
    transformed_raw = F.grid_sample(
        image.raw_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )

    return ImageTensor(raw_tensor=transformed_raw)
