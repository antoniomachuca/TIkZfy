from dataclasses import dataclass
from typing import Any

import torch

from core.dataset.packages import PACKAGE_CATALOG
from core.exceptions import SyntaxTopologicalError, TensorTopologyError

# Whitelisted root environments accepted as the outer drawing environment.
ROOT_ENVIRONMENTS: tuple[str, ...] = ("tikzpicture", "tikzcd", "axis")


@dataclass(frozen=True)
class ImageTensor:
    """
    Immutable image batch tensor of shape (B, C, H, W).
    """
    raw_tensor: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.raw_tensor, torch.Tensor):
            raise TensorTopologyError("Input must be a torch.Tensor instance.")

        # Shape verification: (B, C, H, W)
        if self.raw_tensor.ndim != 4:
            raise TensorTopologyError(
                f"Invalid tensor topology: expected 4D (B,C,H,W), got {self.raw_tensor.ndim}D"
            )

@dataclass(frozen=True)
class TikzTokens:
    """
    Immutable TikZ markup that must contain a supported root environment.

    Root environments are whitelisted to ``tikzpicture`` (canonical training
    environment), ``tikzcd`` (commutative diagrams) and ``axis`` (pgfplots).
    Declared package dependencies are validated against the package catalog so
    compilation can resolve the correct preamble.
    """
    markup: str
    packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.markup, str):
            raise SyntaxTopologicalError("Markup must be a string sequence.")

        stripped_markup = self.markup.strip()
        if not stripped_markup:
            raise SyntaxTopologicalError("Generative markup cannot be an empty sequence.")

        if not any(
            f"\\begin{{{environment}}}" in stripped_markup
            for environment in ROOT_ENVIRONMENTS
        ):
            raise SyntaxTopologicalError(
                "Markup sequence lacks a supported root environment "
                f"({', '.join(ROOT_ENVIRONMENTS)})."
            )

        self._validate_packages(self.packages)

    @staticmethod
    def _validate_packages(packages: tuple[str, ...]) -> None:
        """
        Validates declared package dependencies against the catalog.

        Raises:
            SyntaxTopologicalError: If a package is unknown or not a string.
        """
        if not isinstance(packages, tuple):
            raise SyntaxTopologicalError("Packages must be a tuple of package names.")
        for package in packages:
            if not isinstance(package, str) or package not in PACKAGE_CATALOG:
                raise SyntaxTopologicalError(
                    f"Unknown package '{package}'. Known packages: "
                    f"{tuple(PACKAGE_CATALOG)}."
                )

@dataclass(frozen=True)
class CompilationResult:
    """
    Immutable product of the external TeX compilation sub-process.
    """
    pdf_data: bytes
    is_successful: bool

@dataclass(frozen=True)
class RawLatexDocument:
    """
    Immutable unparsed LaTeX source text.
    """
    raw_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str):
            raise SyntaxTopologicalError("Raw document payload must be a pure string.")

@dataclass(frozen=True)
class TrainingCheckpoint:
    """
    Immutable snapshot of model and optimizer state at a training step.

    ``model_state`` is the ``nn.Module.state_dict()`` mapping parameter names
    to tensors; ``optimizer_state`` is the ``Optimizer.state_dict()`` mapping
    holding per-parameter moments and hyperparameter groups.

    Spatial complexity: O(P) where P is the number of model parameters.
    """
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_state, dict) or not self.model_state:
            raise TensorTopologyError("Model state must be a non-empty state-dict mapping.")
        if not isinstance(self.optimizer_state, dict) or not self.optimizer_state:
            raise TensorTopologyError("Optimizer state must be a non-empty state-dict mapping.")
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise TensorTopologyError("epoch must be a non-negative integer.")
