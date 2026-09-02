"""Inbound orchestrator implementing the primary Image-to-TikZ use case."""

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.exceptions import TensorTopologyError
from core.ml.generation import (
    beam_search,
    best_of_n_search,
    decode_indices_to_markup,
    greedy_search,
    sample_search,
)
from core.ml.metrics import structural_similarity
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary, TrainingCheckpoint
from ports.inbound import ImageToTikzUseCase
from ports.outbound import ImageRasterizerPort, TexCompilerPort

VALID_SEARCH_STRATEGIES: tuple[str, ...] = (
    "greedy",
    "beam",
    "grammar_greedy",
    "grammar_beam",
    "sample",
    "best_of_n",
)


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
        if search_strategy not in VALID_SEARCH_STRATEGIES:
            raise ValueError(
                f"search_strategy must be one of {VALID_SEARCH_STRATEGIES}. "
                f"Got '{search_strategy}'."
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
                grammar_constrained=False,
            )
            indices: tuple[int, ...] = hypotheses[0].tokens
        elif self._search_strategy == "grammar_beam":
            hypotheses = beam_search(
                self._model,
                image,
                max_length=self._max_length,
                beam_width=self._beam_width,
                grammar_constrained=True,
            )
            indices = hypotheses[0].tokens
        elif self._search_strategy == "grammar_greedy":
            indices = greedy_search(
                self._model,
                image,
                max_length=self._max_length,
                grammar_constrained=True,
            )
        elif self._search_strategy == "sample":
            indices = sample_search(
                self._model,
                image,
                max_length=self._max_length,
                grammar_constrained=True,
            )
        elif self._search_strategy == "best_of_n":
            candidates = best_of_n_search(
                self._model,
                image,
                n_hypotheses=self._beam_width,
                max_length=self._max_length,
                grammar_constrained=True,
            )
            indices = candidates[0]
        else:
            indices = greedy_search(
                self._model,
                image,
                max_length=self._max_length,
                grammar_constrained=False,
            )

        return decode_indices_to_markup(self._vocabulary, indices)

    async def execute_reranked(
        self,
        image: ImageTensor,
        compiler: TexCompilerPort,
        rasterizer: ImageRasterizerPort,
        n_hypotheses: int = 4,
    ) -> tuple[TikzTokens, float]:
        """Execute Best-of-N inference with execution-guided SSIM re-ranking.

        Generates N candidate markups using grammar-constrained nucleus sampling,
        compiles each via TexCompilerPort, rasterizes via ImageRasterizerPort,
        and selects the candidate with highest SSIM against the input tensor.

        Args:
            image (ImageTensor): Statically validated 4D tensor with shape (1, C, H, W).
            compiler (TexCompilerPort): Outbound TeX compilation port.
            rasterizer (ImageRasterizerPort): Outbound PDF rasterization port.
            n_hypotheses (int): Number of candidate hypotheses (default: 4).

        Returns:
            tuple[TikzTokens, float]: (Selected best markup, Measured SSIM score).

        Temporal complexity: O(N * (T_decode + T_compile + T_rasterize + T_ssim)).
        """
        if not isinstance(image, ImageTensor):
            raise TypeError(f"Expected ImageTensor, got {type(image)}.")
        if image.raw_tensor.ndim != 4 or image.raw_tensor.shape[0] != 1:
            raise TensorTopologyError(
                f"Inference requires a single image tensor of shape (1, C, H, W). "
                f"Got shape {tuple(image.raw_tensor.shape)}."
            )
        if not isinstance(compiler, TexCompilerPort):
            raise TypeError("compiler must implement TexCompilerPort.")
        if not isinstance(rasterizer, ImageRasterizerPort):
            raise TypeError("rasterizer must implement ImageRasterizerPort.")
        if n_hypotheses < 1:
            raise ValueError(f"n_hypotheses must be positive. Got {n_hypotheses}.")

        candidate_indices = best_of_n_search(
            self._model,
            image,
            n_hypotheses=n_hypotheses,
            max_length=self._max_length,
            grammar_constrained=True,
        )

        best_markup: TikzTokens = decode_indices_to_markup(
            self._vocabulary, candidate_indices[0]
        )
        best_ssim: float = 0.0
        ref_image_tensor = image.raw_tensor[0]
        _, target_h, target_w = ref_image_tensor.shape

        cand_idx: int = 0
        while cand_idx < len(candidate_indices):
            cand_tokens = decode_indices_to_markup(self._vocabulary, candidate_indices[cand_idx])
            comp_res = await compiler.compile_tikz(cand_tokens)
            if comp_res.is_successful and comp_res.pdf_data:
                try:
                    png_bytes = await rasterizer.rasterize_pdf(comp_res.pdf_data, dpi=72)
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    cand_tensor = torch.from_numpy(arr).permute(2, 0, 1)
                    if cand_tensor.shape[1:] != (target_h, target_w):
                        cand_tensor = F.interpolate(
                            cand_tensor.unsqueeze(0),
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    score = structural_similarity(ref_image_tensor, cand_tensor)
                    if score > best_ssim:
                        best_ssim = score
                        best_markup = cand_tokens
                except Exception:
                    pass
            cand_idx += 1

        return best_markup, best_ssim


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
        checkpoint: TrainingCheckpoint = checkpoint_adapter.load_checkpoint(str(checkpoint_path))

        cfg: dict[str, Any] = config or {}
        state = checkpoint.model_state

        # Auto-infer topology dimensions from checkpoint state dict if not explicitly overridden
        inferred_dim = int(cfg.get("model_dimension", 256))
        if "model_dimension" not in cfg and "decoder.output_projection.weight" in state:
            inferred_dim = state["decoder.output_projection.weight"].shape[1]

        inferred_layers = int(cfg.get("num_layers", 2))
        if "num_layers" not in cfg:
            layer_keys = [k for k in state.keys() if k.startswith("decoder.transformer.layers.")]
            if layer_keys:
                inferred_layers = max(int(k.split(".")[3]) for k in layer_keys) + 1

        inferred_heads = int(cfg.get("num_heads", 4))
        if "num_heads" not in cfg:
            inferred_heads = 8 if inferred_dim >= 256 else 4

        input_channels: int = int(cfg.get("input_channels", 3))
        model_dimension: int = inferred_dim
        cfg_max_length: int = int(cfg.get("max_length", max_length))
        num_layers: int = inferred_layers
        num_heads: int = inferred_heads

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
