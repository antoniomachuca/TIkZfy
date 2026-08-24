"""Inbound orchestrator implementing the primary Image-to-TikZ use case."""

from pathlib import Path
from typing import Any

import torch

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.exceptions import TensorTopologyError
from core.ml.generation import beam_search, decode_indices_to_markup, greedy_search
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary, TrainingCheckpoint
from ports.inbound import ImageToTikzUseCase


class ImageToTikzOrchestrator(ImageToTikzUseCase):
    """Orchestrates image-to-markup translation via conditional neural inference.

    Implements the Inbound Port ``ImageToTikzUseCase``, coordinating autoregressive
    decoding and deterministic syntax completion into validated ``TikzTokens``.
    """

    def __init__(
        self,
        model: VisionAutoregressiveModel,
        vocabulary: TokenVocabulary,
        max_length: int = 512,
        search_strategy: str = "greedy",
        beam_width: int = 4,
    ) -> None:
        if not isinstance(model, VisionAutoregressiveModel):
            raise TypeError("model must be a VisionAutoregressiveModel instance.")
        if not isinstance(vocabulary, TokenVocabulary):
            raise TypeError("vocabulary must be a TokenVocabulary instance.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive. Got {max_length}.")
        if search_strategy not in ("greedy", "beam"):
            raise ValueError(
                f"search_strategy must be 'greedy' or 'beam'. Got '{search_strategy}'."
            )
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive. Got {beam_width}.")

        self._model: VisionAutoregressiveModel = model
        self._vocabulary: TokenVocabulary = vocabulary
        self._max_length: int = max_length
        self._search_strategy: str = search_strategy
        self._beam_width: int = beam_width

    @property
    def model(self) -> VisionAutoregressiveModel:
        """Return the underlying vision autoregressive neural model."""
        return self._model

    @property
    def vocabulary(self) -> TokenVocabulary:
        """Return the bound token vocabulary."""
        return self._vocabulary

    @property
    def max_length(self) -> int:
        """Return the maximum decoding sequence budget."""
        return self._max_length

    def execute(self, image: ImageTensor) -> TikzTokens:
        """Execute the generative mapping from spatial tensor to TikZ markup.

        Args:
            image (ImageTensor): Statically validated 4D tensor with shape (1, C, H, W).

        Returns:
            TikzTokens: Decoded and syntax-completed TikZ markup value object.

        Raises:
            TensorTopologyError: If image batch size is not exactly one.
            TypeError: If image is not an ImageTensor.

        Temporal complexity: O(L * D^2) where L is max_length and D is model dimension.
        """
        if not isinstance(image, ImageTensor):
            raise TypeError(f"Expected ImageTensor, got {type(image)}.")
        # Shape verification: (1, C, H, W)
        if image.raw_tensor.ndim != 4 or image.raw_tensor.shape[0] != 1:
            raise TensorTopologyError(
                f"Inference requires a single image tensor of shape (1, C, H, W). "
                f"Got shape {tuple(image.raw_tensor.shape)}."
            )

        if self._search_strategy == "beam":
            hypotheses = beam_search(
                self._model,
                image,
                max_length=self._max_length,
                beam_width=self._beam_width,
            )
            indices: tuple[int, ...] = hypotheses[0].tokens
        else:
            indices = greedy_search(
                self._model,
                image,
                max_length=self._max_length,
            )

        return decode_indices_to_markup(self._vocabulary, indices)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocabulary_path: str | Path,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
        max_length: int = 512,
        search_strategy: str = "greedy",
        beam_width: int = 4,
    ) -> "ImageToTikzOrchestrator":
        """Factory method to construct an orchestrator from persisted artifacts.

        Args:
            checkpoint_path (str | Path): Path to the serialized .pt checkpoint file.
            vocabulary_path (str | Path): Path to the serialized vocabulary.json file.
            config (dict[str, Any] | None): Optional model hyperparameters mapping.
            device (torch.device | None): Computation device (CPU or CUDA).
            max_length (int): Maximum generative sequence length.
            search_strategy (str): 'greedy' or 'beam' decoding.
            beam_width (int): Beam width for beam search.

        Returns:
            ImageToTikzOrchestrator: Fully initialized and evaluated orchestrator.
        """
        vocab_adapter: JsonVocabularyAdapter = JsonVocabularyAdapter()
        vocabulary: TokenVocabulary = vocab_adapter.load_vocabulary(str(vocabulary_path))

        checkpoint_adapter: AtomicCheckpointAdapter = AtomicCheckpointAdapter()
        checkpoint: TrainingCheckpoint = checkpoint_adapter.load_checkpoint(
            str(checkpoint_path)
        )

        cfg: dict[str, Any] = config or {}
        input_channels: int = int(cfg.get("input_channels", 3))
        model_dimension: int = int(cfg.get("model_dimension", 256))
        cfg_max_length: int = int(cfg.get("max_length", max_length))
        num_layers: int = int(cfg.get("num_layers", 2))
        num_heads: int = int(cfg.get("num_heads", 4))

        model: VisionAutoregressiveModel = VisionAutoregressiveModel(
            vocabulary=vocabulary,
            input_channels=input_channels,
            model_dimension=model_dimension,
            max_length=cfg_max_length,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.load_state_dict(checkpoint.model_state)

        if device is not None:
            model.to(device)

        model.eval()

        return cls(
            model=model,
            vocabulary=vocabulary,
            max_length=cfg_max_length,
            search_strategy=search_strategy,
            beam_width=beam_width,
        )


class DemoImageToTikzOrchestrator(ImageToTikzUseCase):
    """Deterministic demonstration orchestrator for end-to-end UI verification.

    Generates mathematically valid, compile-ready TikZ figures when neural checkpoints
    are pending execution or during development validation.
    """

    _DEMO_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            (
                r"\begin{tikzpicture}[scale=1.2] "
                r"\draw[thick, fill=blue!15] (0,0) -- (4,0) -- (2,3) -- cycle; "
                r"\draw[dashed, red] (2,0) -- (2,3); "
                r"\node at (2,-0.3) {base $b$}; "
                r"\node at (2.4,1.5) {height $h$}; "
                r"\node at (2,0.8) {Area $= \frac{1}{2}bh$}; "
                r"\end{tikzpicture}"
            ),
            (),
        ),
        (
            (
                r"\begin{tikzpicture}[scale=1.0] "
                r"\draw[->, thick] (-2.5,0) -- (2.5,0) node[right] {$x$}; "
                r"\draw[->, thick] (0,-2.5) -- (0,2.5) node[above] {$y$}; "
                r"\draw[thick, blue, fill=blue!10] (0,0) circle (1.8); "
                r"\draw[->, red, thick] (0,0) -- (1.27,1.27) node[midway, above left] {$r$}; "
                r"\filldraw[black] (0,0) circle (1.5pt) node[below left] {$O$}; "
                r"\node at (0,-1.0) {$x^2 + y^2 = r^2$}; "
                r"\end{tikzpicture}"
            ),
            (),
        ),
        (
            (
                r"\begin{tikzpicture}[scale=1.1] "
                r"\draw[thick, domain=-2:2, samples=50, color=red] plot (\x, {0.5*\x*\x - 1}) "
                r"node[right] {$f(x) = \frac{1}{2}x^2 - 1$}; "
                r"\draw[->] (-2.5,0) -- (2.5,0) node[right] {$x$}; "
                r"\draw[->] (0,-1.8) -- (0,2.2) node[above] {$y$}; "
                r"\filldraw[blue] (0,-1) circle (2pt) node[below right] {$(0, -1)$}; "
                r"\end{tikzpicture}"
            ),
            (),
        ),
        (
            (
                r"\begin{tikzpicture}[node distance=2.5cm, auto] "
                r"\node[circle, draw=black, thick, fill=gray!20] (A) {$X$}; "
                r"\node[circle, draw=black, thick, fill=gray!20] (B) [right of=A] {$Y$}; "
                r"\node[circle, draw=black, thick, fill=gray!20] (C) [below of=B] {$Z$}; "
                r"\draw[->, thick] (A) to node {$f$} (B); "
                r"\draw[->, thick] (B) to node {$g$} (C); "
                r"\draw[->, dashed, thick, blue] (A) to node [below left] {$g \circ f$} (C); "
                r"\end{tikzpicture}"
            ),
            (),
        ),
    )

    def execute(self, image: ImageTensor) -> TikzTokens:
        """Deterministically map spatial image statistics to a valid TikZ geometry.

        Args:
            image (ImageTensor): Input tensor with shape (1, C, H, W).

        Returns:
            TikzTokens: Validated compile-ready TikZ token sequence.
        """
        if not isinstance(image, ImageTensor):
            raise TypeError("Expected ImageTensor instance.")
        # Shape verification: (1, C, H, W)
        mean_val: float = float(image.raw_tensor.mean().item())
        index: int = int(abs(mean_val * 1000)) % len(self._DEMO_TEMPLATES)
        markup, packages = self._DEMO_TEMPLATES[index]
        return TikzTokens(markup=markup, packages=packages)


__all__ = ["DemoImageToTikzOrchestrator", "ImageToTikzOrchestrator"]
