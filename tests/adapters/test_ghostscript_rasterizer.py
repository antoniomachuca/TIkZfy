import asyncio

import pytest

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.exceptions import DomainError
from core.models import TikzTokens

PNG_MAGIC_BYTES: bytes = b"\x89PNG\r\n\x1a\n"

MINIMAL_MARKUP: str = (
    "\\begin{tikzpicture}\n"
    "\\draw (0,0) -- (1,1);\n"
    "\\end{tikzpicture}"
)


@pytest.mark.infrastructure
def test_rasterize_compiled_tikz_produces_png() -> None:
    """Verify the compile-rasterize chain yields a PNG artifact."""

    async def _pipeline() -> bytes:
        compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")
        compilation = await compiler.compile_tikz(TikzTokens(markup=MINIMAL_MARKUP))
        rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()
        return await rasterizer.rasterize_pdf(compilation.pdf_data)

    png_data: bytes = asyncio.run(_pipeline())
    assert png_data[:8] == PNG_MAGIC_BYTES
    assert len(png_data) > 0


@pytest.mark.infrastructure
def test_rasterize_invalid_pdf_raises_domain_error() -> None:
    """Verify corrupt payloads are mapped to DomainError."""

    async def _pipeline() -> bytes:
        rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()
        return await rasterizer.rasterize_pdf(b"%PDF-corrupted-payload")

    with pytest.raises(DomainError):
        asyncio.run(_pipeline())


@pytest.mark.infrastructure
def test_rasterize_rejects_non_positive_dpi() -> None:
    """Verify the dpi guard clause executes before any subprocess spawn."""
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()
    with pytest.raises(DomainError):
        asyncio.run(rasterizer.rasterize_pdf(b"%PDF-1.4", dpi=0))
