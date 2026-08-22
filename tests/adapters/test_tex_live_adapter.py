import asyncio

import pytest

from adapters.tex_live_adapter import (
    AsyncTexLiveAdapter,
    categorize_compilation_failure,
)
from core.exceptions import CompilationSyntaxError, MissingPackageError
from core.models import TikzTokens


def test_categorize_missing_package_failure() -> None:
    """Verify a missing ``.sty`` log maps to a MissingPackageError with the name."""
    log = "! LaTeX Error: File `pgfplots.sty' not found.\nSee the LaTeX manual."
    error = categorize_compilation_failure(log)
    assert isinstance(error, MissingPackageError)
    assert "pgfplots" in str(error)


def test_categorize_syntax_failure() -> None:
    """Verify a generic failure log maps to a CompilationSyntaxError."""
    log = "! Undefined control sequence.\nl.6 \\begin{axis}"
    error = categorize_compilation_failure(log)
    assert isinstance(error, CompilationSyntaxError)


def test_categorize_missing_class_file() -> None:
    """Verify a missing ``.cls`` file is categorized as a missing package."""
    log = "! LaTeX Error: File `standalone.cls' not found."
    error = categorize_compilation_failure(log)
    assert isinstance(error, MissingPackageError)
    assert "standalone" in str(error)


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    ("markup", "packages"),
    [
        (
            "\\begin{tikzpicture}\\begin{axis}"
            "\\addplot coordinates {(0,0) (1,1) (2,1.5)};\\end{axis}\\end{tikzpicture}",
            ("pgfplots",),
        ),
        (
            "\\begin{tikzcd} A \\arrow[r] & B \\end{tikzcd}",
            ("tikz-cd",),
        ),
        (
            "\\begin{tikzpicture}\\draw (0,0) to[R, l=$R_1$] (2,0);\\end{tikzpicture}",
            ("circuitikz",),
        ),
    ],
)
def test_compile_package_aware_markup(markup: str, packages: tuple[str, ...]) -> None:
    """Verify package-aware markups compile with the catalog-resolved preamble."""

    async def _compile() -> bool:
        compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")
        result = await compiler.compile_tikz(TikzTokens(markup=markup, packages=packages))
        return result.is_successful and len(result.pdf_data) > 0

    assert asyncio.run(_compile()) is True
