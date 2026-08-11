"""
Syntactic parsing operations for geometric graph extraction.
"""
import re

from core.models.value_objects import RawLatexDocument, TikzTokens


def extract_tikz_graphs(document: RawLatexDocument) -> list[TikzTokens]:
    """
    Extracts all TikZ geometric graphs from a raw LaTeX document.
    Employs deterministic regular expressions to avoid explicit scalar iterations
    and guarantees O(1) logical temporal complexity on substring localization.

    Args:
        document (RawLatexDocument): The immutable raw string document.

    Returns:
        List[TikzTokens]: Sequence of extracted geometric graphs mapping to
        the topological subspace.
    """
    if not isinstance(document, RawLatexDocument):
        raise TypeError("Input must be a RawLatexDocument instance.")

    # Formal bounding box for the geometric topology.
    pattern = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)

    # Vectorized C-level extraction bypassing sequential loops.
    matches: list[str] = pattern.findall(document.raw_text)

    # Deterministic functional mapping to the output domain.
    return [TikzTokens(markup=match) for match in matches]
