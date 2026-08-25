from pathlib import Path

import pytest
import torch

from adapters.tensor_persistence import PyTorchTensorAdapter
from core.exceptions import DomainError
from core.models.value_objects import ImageTensor


@pytest.fixture
def tensor_adapter() -> PyTorchTensorAdapter:
    return PyTorchTensorAdapter()


@pytest.fixture
def valid_image_tensor() -> ImageTensor:
    # Shape: (B, C, H, W)
    raw = torch.randn(2, 3, 64, 64)
    return ImageTensor(raw_tensor=raw)


def test_save_and_load_tensor_success(
    tensor_adapter: PyTorchTensorAdapter, valid_image_tensor: ImageTensor, tmp_path: Path
) -> None:
    """
    Saving then loading preserves the tensor values.
    """
    file_path = str(tmp_path / "test_tensor.pt")

    # Save the tensor
    tensor_adapter.save_tensor(valid_image_tensor, file_path)

    # Verify file exists
    assert Path(file_path).exists()

    # Load the tensor
    loaded_tensor = tensor_adapter.load_tensor(file_path)

    # Verify structural integrity
    assert isinstance(loaded_tensor, ImageTensor)
    assert loaded_tensor.raw_tensor.shape == valid_image_tensor.raw_tensor.shape
    assert torch.equal(loaded_tensor.raw_tensor, valid_image_tensor.raw_tensor)


def test_save_tensor_invalid_type(tensor_adapter: PyTorchTensorAdapter, tmp_path: Path) -> None:
    """
    Saving a non-ImageTensor raises DomainError.
    """
    file_path = str(tmp_path / "invalid_save.pt")

    # We purposefully pass an invalid type (e.g., a raw string or dict)
    with pytest.raises(DomainError, match="Input must be an ImageTensor instance."):
        tensor_adapter.save_tensor("not_a_tensor", file_path)  # type: ignore


def test_load_tensor_nonexistent_file(tensor_adapter: PyTorchTensorAdapter) -> None:
    """
    Loading from a missing file raises DomainError.
    """
    with pytest.raises(DomainError, match="Source path does not exist"):
        tensor_adapter.load_tensor("/path/that/does/not/exist.pt")


def test_load_tensor_invalid_payload(tensor_adapter: PyTorchTensorAdapter, tmp_path: Path) -> None:
    """
    Loading a corrupted or non-tensor payload raises DomainError.
    """
    file_path = str(tmp_path / "invalid_payload.pt")

    # Create a dummy file that is not a valid PyTorch tensor payload
    with open(file_path, "w") as f:
        f.write("corrupted data")

    with pytest.raises(DomainError, match="Failed to load tensor"):
        tensor_adapter.load_tensor(file_path)


def test_load_tensor_invalid_topology(tensor_adapter: PyTorchTensorAdapter, tmp_path: Path) -> None:
    """
    Loading a tensor with invalid dimensions (2D instead of 4D) raises DomainError.
    """
    file_path = str(tmp_path / "invalid_topology.pt")

    # Shape: (H, W) - Invalid topology for ImageTensor
    invalid_raw = torch.randn(64, 64)
    torch.save(invalid_raw, file_path)

    with pytest.raises(DomainError, match="violates structural invariants"):
        tensor_adapter.load_tensor(file_path)
