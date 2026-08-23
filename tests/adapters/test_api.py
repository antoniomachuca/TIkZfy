"""Unit and integration tests for the FastAPI HTTP adapter layer."""

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from adapters.api.app import create_app
from adapters.api.image_decoder import decode_image_bytes_to_tensor
from core.models import CompilationResult, ImageTensor, TikzTokens
from ports.inbound import ImageToTikzUseCase
from ports.outbound import ImageRasterizerPort, TexCompilerPort


class StubOrchestrator(ImageToTikzUseCase):
    """Stub orchestrator returning deterministic TikZ markup."""

    def __init__(
        self,
        markup: str = r"\begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture}",
    ) -> None:
        self.markup = markup

    def execute(self, image: ImageTensor) -> TikzTokens:
        if not isinstance(image, ImageTensor):
            raise TypeError("Expected ImageTensor")
        return TikzTokens(markup=self.markup, packages=())


class StubCompiler(TexCompilerPort):
    """Stub compiler returning mock compilation result."""

    def __init__(self, succeeds: bool = True) -> None:
        self.succeeds = succeeds

    async def compile_tikz(self, tokens: TikzTokens) -> CompilationResult:
        if self.succeeds:
            return CompilationResult(
                pdf_data=b"%PDF-1.4 mock pdf data",
                is_successful=True,
            )
        return CompilationResult(
            pdf_data=b"",
            is_successful=False,
        )


class StubRasterizer(ImageRasterizerPort):
    """Stub rasterizer returning mock PNG bytes."""

    async def rasterize_pdf(self, pdf_data: bytes, dpi: int = 150) -> bytes:
        return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR mock png"


def _create_synthetic_png_bytes(width: int = 64, height: int = 64, mode: str = "RGB") -> bytes:
    """Generate in-memory PNG bytes for testing."""
    image = Image.new(mode, (width, height), color=(128, 128, 128) if mode != "L" else 128)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_bytes_to_tensor_rgb() -> None:
    png_bytes = _create_synthetic_png_bytes(64, 64, "RGB")
    tensor = decode_image_bytes_to_tensor(png_bytes, target_height=32, target_width=32)

    assert isinstance(tensor, ImageTensor)
    assert tensor.raw_tensor.shape == (1, 3, 32, 32)
    assert tensor.raw_tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.raw_tensor.min()) <= float(tensor.raw_tensor.max()) <= 1.0


def test_decode_image_bytes_to_tensor_grayscale() -> None:
    png_bytes = _create_synthetic_png_bytes(64, 64, "L")
    tensor = decode_image_bytes_to_tensor(png_bytes, target_height=32, target_width=32)

    assert isinstance(tensor, ImageTensor)
    assert tensor.raw_tensor.shape == (1, 3, 32, 32)


def test_decode_image_bytes_to_tensor_rgba() -> None:
    png_bytes = _create_synthetic_png_bytes(64, 64, "RGBA")
    tensor = decode_image_bytes_to_tensor(png_bytes, target_height=32, target_width=32)

    assert isinstance(tensor, ImageTensor)
    assert tensor.raw_tensor.shape == (1, 3, 32, 32)


def test_decode_image_bytes_to_tensor_validation() -> None:
    with pytest.raises(ValueError):
        decode_image_bytes_to_tensor(b"")
    with pytest.raises(ValueError):
        decode_image_bytes_to_tensor(b"invalid_bytes_not_an_image")
    with pytest.raises(ValueError):
        decode_image_bytes_to_tensor(_create_synthetic_png_bytes(), target_height=0)


def test_health_endpoint() -> None:
    app = create_app(orchestrator=StubOrchestrator())
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert data["model_loaded"] is True


def test_health_endpoint_no_model() -> None:
    app = create_app(orchestrator=None)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_generate_endpoint_success() -> None:
    orchestrator = StubOrchestrator(
        markup=r"\begin{tikzpicture} \draw (0,0) circle (1); \end{tikzpicture}"
    )
    compiler = StubCompiler(succeeds=True)
    rasterizer = StubRasterizer()
    app = create_app(orchestrator=orchestrator, compiler=compiler, rasterizer=rasterizer)
    client = TestClient(app)

    png_bytes = _create_synthetic_png_bytes(64, 64)
    files = {"image": ("test.png", png_bytes, "image/png")}

    response = client.post("/api/v1/generate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "tikz_code" in data
    assert r"\begin{tikzpicture}" in data["tikz_code"]
    assert "packages" in data
    assert data["packages"] == []
    assert data["compilation_success"] is True
    assert data["preview_url"] is not None
    assert data["preview_url"].startswith("data:image/png;base64,")


def test_generate_endpoint_empty_file_fails() -> None:
    app = create_app(orchestrator=StubOrchestrator())
    client = TestClient(app)

    files = {"image": ("empty.png", b"", "image/png")}
    response = client.post("/api/v1/generate", files=files)
    assert response.status_code == 400


def test_generate_endpoint_invalid_image_fails() -> None:
    app = create_app(orchestrator=StubOrchestrator())
    client = TestClient(app)

    files = {"image": ("corrupted.png", b"this is not an image", "image/png")}
    response = client.post("/api/v1/generate", files=files)
    assert response.status_code == 400


def test_generate_endpoint_no_orchestrator_returns_503() -> None:
    app = create_app(orchestrator=None)
    client = TestClient(app)

    png_bytes = _create_synthetic_png_bytes(64, 64)
    files = {"image": ("test.png", png_bytes, "image/png")}
    response = client.post("/api/v1/generate", files=files)
    assert response.status_code == 503


def test_compile_endpoint_success() -> None:
    compiler = StubCompiler(succeeds=True)
    rasterizer = StubRasterizer()
    app = create_app(compiler=compiler, rasterizer=rasterizer)
    client = TestClient(app)

    payload = {
        "tikz_code": r"\begin{tikzpicture} \draw (0,0) -- (2,2); \end{tikzpicture}",
        "packages": [],
    }
    response = client.post("/api/v1/compile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["preview_url"] is not None
    assert data["preview_url"].startswith("data:image/png;base64,")


def test_compile_endpoint_failure() -> None:
    compiler = StubCompiler(succeeds=False)
    rasterizer = StubRasterizer()
    app = create_app(compiler=compiler, rasterizer=rasterizer)
    client = TestClient(app)

    payload = {
        "tikz_code": r"\invalid command",
        "packages": [],
    }
    response = client.post("/api/v1/compile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["preview_url"] is None
    assert data["error"] is not None
