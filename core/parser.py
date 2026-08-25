"""
Syntactic parsing operations for geometric graph extraction.
"""

import re

from core.models.value_objects import RawLatexDocument, TikzTokens


def extract_tikz_graphs(document: RawLatexDocument) -> list[TikzTokens]:
    """
    Extracts every `tikzpicture` block from a raw LaTeX document.

    Uses a single compiled regex over the whole document.

    Args:
        document (RawLatexDocument): The raw document text.

    Returns:
        list[TikzTokens]: One TikzTokens entry per matched block.

    Temporal complexity: O(D) where D is the document length.
    """
    if not isinstance(document, RawLatexDocument):
        raise TypeError("Input must be a RawLatexDocument instance.")

    # Match each tikzpicture block (non-greedy, DOTALL).
    pattern = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)

    # findall runs in C; no Python loop needed.
    matches: list[str] = pattern.findall(document.raw_text)

    # Wrap each match as a TikzTokens value object.
    return [TikzTokens(markup=match) for match in matches]
