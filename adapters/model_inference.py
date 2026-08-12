"""Concrete model inference adapter for the outbound inference port."""

from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens
from ports.outbound import ModelInferencePort


class TorchModelInferenceAdapter(ModelInferencePort):
    """Expose the pure PyTorch model through ``ModelInferencePort``."""

    def __init__(self, model: VisionAutoregressiveModel) -> None:
        if not isinstance(model, VisionAutoregressiveModel):
            raise TypeError("model must be a VisionAutoregressiveModel instance.")
        self._model: VisionAutoregressiveModel = model

    def infer_markup(self, image: ImageTensor) -> TikzTokens:
        """Delegate bounded autoregressive inference to the domain model."""
        return self._model.generate_markup(image)
