import torch
import torchvision.io as io

from core.models.value_objects import ImageTensor
from ports.outbound import ImageLoaderPort


class TorchVisionImageLoader(ImageLoaderPort):
    """
    Infrastructural adapter for direct 2D image loading via torchvision.
    Isolates file I/O and format decoding from the pure mathematical domain.
    """

    def load_image(self, source_path: str) -> ImageTensor:
        """
        Loads an image directly into RAM and unsqueezes the batch dimension.

        Args:
            source_path (str): The absolute path to the image resource.

        Returns:
            ImageTensor: The domain tensor constrained to shape (B, C, H, W).
        """
        # Shape: (C, H, W)
        raw_tensor = io.read_image(source_path)
        
        float_tensor = raw_tensor.to(dtype=torch.float32)

        # O(1) Batch dimension injection. Shape: (B, C, H, W) where B = 1
        batched_tensor = float_tensor.unsqueeze(0)

        return ImageTensor(raw_tensor=batched_tensor)
