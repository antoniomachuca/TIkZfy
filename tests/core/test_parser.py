import pytest

from core.models.value_objects import RawLatexDocument, TikzTokens
from core.parser import extract_tikz_graphs


def test_extract_tikz_graphs_success() -> None:
    """
    Validates deterministic extraction of geometric graphs from a string mapping.
    """
    raw_str = (
        "Some text \\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture} "
        "More text \\begin{tikzpicture} \\node {A}; \\end{tikzpicture}"
    )
    document = RawLatexDocument(raw_text=raw_str)

    tokens = extract_tikz_graphs(document)

    assert len(tokens) == 2
    assert isinstance(tokens[0], TikzTokens)
    assert "\\draw (0,0) -- (1,1);" in tokens[0].markup
    assert "\\node {A};" in tokens[1].markup


def test_extract_tikz_graphs_no_match() -> None:
    """
    Ensures an empty sequence is mapped when the topological bound is absent.
    """
    raw_str = "This document contains no geometric graphs."
    document = RawLatexDocument(raw_text=raw_str)

    tokens = extract_tikz_graphs(document)

    assert len(tokens) == 0


def test_extract_tikz_graphs_type_error() -> None:
    """
    Confirms type invariance when receiving invalid parameters.
    """
    with pytest.raises(TypeError):
        extract_tikz_graphs("Invalid string payload")  # type: ignore
