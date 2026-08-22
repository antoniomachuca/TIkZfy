from dataclasses import FrozenInstanceError

import pytest
import torch

from core.exceptions import SyntaxTopologicalError, TensorTopologyError
from core.models import CompilationResult, ImageTensor, TikzTokens


def test_image_tensor_valid_topology() -> None:
    """Verify that a valid (B, C, H, W) tensor instantiates without O(N) iteration."""
    # Shape: (1, 3, 256, 256)
    valid_tensor = torch.zeros(1, 3, 256, 256)
    image = ImageTensor(raw_tensor=valid_tensor)
    assert image.raw_tensor.shape == (1, 3, 256, 256)


def test_image_tensor_invalid_type() -> None:
    """Verify strict type constraints reject non-tensor structures."""
    with pytest.raises(TensorTopologyError):
        ImageTensor(raw_tensor=[0, 0, 0])  # type: ignore


def test_image_tensor_invalid_dimensions() -> None:
    """Verify spatial dimensionality constraints in O(1)."""
    # Shape: (3, 256, 256) -> Missing Batch dimension
    invalid_tensor = torch.zeros(3, 256, 256)
    with pytest.raises(TensorTopologyError):
        ImageTensor(raw_tensor=invalid_tensor)


def test_image_tensor_immutability() -> None:
    """Verify state immutability mapping to @dataclass(frozen=True)."""
    valid_tensor = torch.zeros(1, 3, 256, 256)
    image = ImageTensor(raw_tensor=valid_tensor)

    with pytest.raises(FrozenInstanceError):
        image.raw_tensor = torch.zeros(1, 3, 512, 512)  # type: ignore


def test_tikz_tokens_valid_syntax() -> None:
    """Verify that a syntactically bounded sequence instantiates successfully."""
    valid_markup = "\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}"
    tokens = TikzTokens(markup=valid_markup)
    assert tokens.markup == valid_markup


def test_tikz_tokens_empty_sequence() -> None:
    """Verify empty sequences raise structural exceptions."""
    with pytest.raises(SyntaxTopologicalError):
        TikzTokens(markup="   \n  ")


def test_tikz_tokens_missing_bounds() -> None:
    """Verify missing topological bounds (tikzpicture) raises exceptions."""
    invalid_markup = "\\draw (0,0) -- (1,1);"
    with pytest.raises(SyntaxTopologicalError):
        TikzTokens(markup=invalid_markup)


def test_tikz_tokens_accepts_tikzcd_root_environment() -> None:
    """Verify the tikzcd root environment is accepted by the whitelist."""
    markup = "\\begin{tikzcd} A \\arrow[r] & B \\end{tikzcd}"
    tokens = TikzTokens(markup=markup)
    assert tokens.markup == markup


def test_tikz_tokens_accepts_axis_root_environment() -> None:
    """Verify the axis root environment is accepted by the whitelist."""
    markup = "\\begin{axis}\\addplot coordinates {(0,0) (1,1)};\\end{axis}"
    tokens = TikzTokens(markup=markup)
    assert tokens.markup == markup


def test_tikz_tokens_default_packages_empty() -> None:
    """Verify the packages field defaults to an empty tuple."""
    tokens = TikzTokens(markup="\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}")
    assert tokens.packages == ()


def test_tikz_tokens_valid_packages() -> None:
    """Verify declared packages are retained when present in the catalog."""
    markup = "\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}"
    tokens = TikzTokens(markup=markup, packages=("pgfplots", "tikz-cd"))
    assert tokens.packages == ("pgfplots", "tikz-cd")


def test_tikz_tokens_unknown_package_raises() -> None:
    """Verify packages absent from the catalog raise structural exceptions."""
    markup = "\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}"
    with pytest.raises(SyntaxTopologicalError):
        TikzTokens(markup=markup, packages=("nonexistent",))


def test_tikz_tokens_non_tuple_packages_raises() -> None:
    """Verify non-tuple packages declarations raise structural exceptions."""
    markup = "\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}"
    with pytest.raises(SyntaxTopologicalError):
        TikzTokens(markup=markup, packages=["pgfplots"])  # type: ignore


def test_compilation_result_instantiation() -> None:
    """Verify payload mapping for compilation products."""
    payload = b"%PDF-1.4\n..."
    result = CompilationResult(pdf_data=payload, is_successful=True)
    assert result.pdf_data == payload
    assert result.is_successful is True
