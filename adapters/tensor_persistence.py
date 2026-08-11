from pathlib import Path

import torch

from core.exceptions import DomainError, TensorTopologyError
from core.models.value_objects import ImageTensor
from ports.outbound import TensorPersistencePort


class PyTorchTensorAdapter(TensorPersistencePort):
    """
    PyTorch adapter implementing TensorPersistencePort.

    Uses native PyTorch formats (.pt/.pth). Loads use mmap so tensor data is
    mapped from disk lazily instead of being read fully into memory.
    """

    def save_tensor(self, tensor: ImageTensor, destination_path: str) -> None:
        """
        Serializes an ImageTensor to disk.

        Args:
            tensor (ImageTensor): Tensor to save.
            destination_path (str): Path where the tensor is written.

        Raises:
            DomainError: If the file cannot be written.
        """
        if not isinstance(tensor, ImageTensor):
            raise DomainError("Input must be an ImageTensor instance.")

        try:
            path = Path(destination_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor.raw_tensor, path)
        except (OSError, TypeError, RuntimeError) as e:
            raise DomainError(f"Failed to save tensor to '{destination_path}': {str(e)}") from e

    def load_tensor(self, source_path: str) -> ImageTensor:
        """
        Loads a tensor from disk.

        Uses mmap=True so data is mapped from disk rather than fully loaded into RAM.

        Args:
            source_path (str): Path of the stored tensor.

        Returns:
            ImageTensor: The loaded tensor.

        Raises:
            DomainError: If the file is missing, unreadable, or not a valid tensor.
        """
        try:
            path = Path(source_path)
            if not path.exists():
                raise DomainError(f"Source path does not exist: '{source_path}'")

            # weights_only=True avoids executing arbitrary code; mmap=True defers loading.
            raw_tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

            if not isinstance(raw_tensor, torch.Tensor):
                raise DomainError("Loaded payload is not a valid torch.Tensor.")

            return ImageTensor(raw_tensor=raw_tensor)

        except (OSError, RuntimeError) as e:
            raise DomainError(f"Failed to load tensor from '{source_path}': {str(e)}") from e
        except TensorTopologyError as e:
            raise DomainError(f"Loaded tensor violates structural invariants: {str(e)}") from e
