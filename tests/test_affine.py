import pytest
import torch

from core.models.value_objects import ImageTensor
from core.math.affine import apply_affine_transformation
from core.exceptions import TensorTopologyError


def test_affine_transformation_valid():
    """
    Validates that a correctly shaped affine transformation executes
    without spatial iteration logic and maintains topology.
    """
    # Simulate a (1, 3, 256, 256) tensor
    raw = torch.randn(1, 3, 256, 256)
    image = ImageTensor(raw_tensor=raw)
    
    # Identity transformation matrix: Shape (1, 2, 3)
    theta = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]]])
    
    transformed_image = apply_affine_transformation(image, theta)
    
    assert isinstance(transformed_image, ImageTensor)
    assert transformed_image.raw_tensor.shape == (1, 3, 256, 256)
    
    # Mathematical assertion: Identity transform should leave the tensor approximately unchanged
    # Grid sampling introduces slight interpolation numerical errors, so we use allclose
    assert torch.allclose(raw, transformed_image.raw_tensor, atol=1e-4)


def test_affine_transformation_invalid_theta_shape():
    """
    Validates that non-compliant affine matrices are rejected at O(1) boundaries.
    """
    raw = torch.randn(1, 3, 256, 256)
    image = ImageTensor(raw_tensor=raw)
    
    # Invalid shape (1, 3, 3) instead of (1, 2, 3)
    theta = torch.randn(1, 3, 3)
    
    with pytest.raises(TensorTopologyError):
        apply_affine_transformation(image, theta)


def test_affine_transformation_batch_mismatch():
    """
    Validates that batch discrepancies between image and theta are structurally rejected.
    """
    raw = torch.randn(2, 3, 256, 256)
    image = ImageTensor(raw_tensor=raw)
    
    # Batch size of 1 for theta vs 2 for image
    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    
    with pytest.raises(TensorTopologyError):
        apply_affine_transformation(image, theta)
