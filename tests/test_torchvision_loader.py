import os
import tempfile

import torch

from adapters.torchvision_loader import TorchVisionImageLoader
from core.models.value_objects import ImageTensor


def test_torchvision_loader_success() -> None:
    """
    Validates that the infrastructural adapter successfully loads an image
    into the O(1) mathematical domain representation without sequential iteration.
    """
    loader = TorchVisionImageLoader()

    # Create a dummy image using torch directly for isolated infrastructural testing
    dummy_tensor = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        import torchvision.io as io

        io.write_png(dummy_tensor, tmp.name)
        tmp_path = tmp.name

    try:
        # Action
        image = loader.load_image(tmp_path)

        # Verification
        assert isinstance(image, ImageTensor)
        assert image.raw_tensor.shape == (1, 3, 64, 64)
        assert image.raw_tensor.dtype == torch.float32

    finally:
        os.remove(tmp_path)
