"""FastAPI application factory and REST endpoints for the Image-to-TikZ engine."""

import base64
import os
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from adapters.api.image_decoder import decode_image_bytes_to_tensor
from adapters.api.schemas import (
    CompileRequest,
    CompileResponse,
    GenerateResponse,
    HealthResponse,
)
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.orchestrator import ImageToTikzOrchestrator
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.exceptions import DomainError
from core.models import ImageTensor, TikzTokens
from ports.inbound import ImageToTikzUseCase
from ports.outbound import ImageRasterizerPort, TexCompilerPort


def create_app(
    orchestrator: ImageToTikzUseCase | None = None,
    compiler: TexCompilerPort | None = None,
    rasterizer: ImageRasterizerPort | None = None,
    target_height: int = 224,
    target_width: int = 224,
    enable_cors: bool = True,
) -> FastAPI:
    """Construct and configure the FastAPI application.

    Args:
        orchestrator (ImageToTikzUseCase | None): Inbound port orchestrator.
        compiler (TexCompilerPort | None): Outbound port LaTeX compiler.
        rasterizer (ImageRasterizerPort | None): Outbound port PDF rasterizer.
        target_height (int): Input spatial height for tensor ingestion.
        target_width (int): Input spatial width for tensor ingestion.
        enable_cors (bool): Whether to attach CORS middleware.

    Returns:
        FastAPI: Fully configured ASGI application.
    """
    app: FastAPI = FastAPI(
        title="Image-to-TikZ Neural Engine API",
        description=(
            "Multimodal neural engine for translating figures into compile-ready TikZ LaTeX code."
        ),
        version="0.1.0",
    )

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Resolution of dependencies
    active_compiler: TexCompilerPort = compiler or AsyncTexLiveAdapter()
    active_rasterizer: ImageRasterizerPort = rasterizer or GhostscriptRasterizer()
    active_orchestrator: ImageToTikzUseCase | None = orchestrator

    if active_orchestrator is None:
        checkpoint_env: str = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
        vocab_env: str = os.getenv(
            "VOCABULARY_PATH", "dataset/encoded/vocabulary.json"
        )
        if Path(checkpoint_env).exists() and Path(vocab_env).exists():
            active_orchestrator = ImageToTikzOrchestrator.from_checkpoint(
                checkpoint_path=checkpoint_env,
                vocabulary_path=vocab_env,
            )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service and device telemetry health check",
    )
    async def health() -> HealthResponse:
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
        return HealthResponse(
            status="ok",
            device=device_str,
            model_loaded=active_orchestrator is not None,
        )

    @app.post(
        "/api/v1/generate",
        response_model=GenerateResponse,
        summary="Generate TikZ markup and rasterized preview from an uploaded image",
    )
    async def generate_tikz(
        image: Annotated[UploadFile, File(description="Image file (PNG, JPEG, WebP)")],
    ) -> GenerateResponse:
        if active_orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neural inference orchestrator is not initialized.",
            )

        payload_bytes: bytes = await image.read()
        if not payload_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file received.",
            )

        image_tensor: ImageTensor
        try:
            image_tensor = decode_image_bytes_to_tensor(
                image_bytes=payload_bytes,
                target_height=target_height,
                target_width=target_width,
            )
        except (ValueError, DomainError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format or tensor topology: {str(exc)}",
            ) from exc

        tokens: TikzTokens
        try:
            tokens = active_orchestrator.execute(image_tensor)
        except DomainError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Generative inference failed: {str(exc)}",
            ) from exc

        # Attempt LaTeX compilation and rasterization for preview
        preview_url: str | None = None
        compilation_success: bool = True
        try:
            compilation_result = await active_compiler.compile_tikz(tokens)
            if compilation_result.is_successful and compilation_result.pdf_data:
                png_bytes = await active_rasterizer.rasterize_pdf(
                    compilation_result.pdf_data
                )
                b64_str = base64.b64encode(png_bytes).decode("ascii")
                preview_url = f"data:image/png;base64,{b64_str}"
            else:
                compilation_success = False
        except Exception:
            compilation_success = False

        return GenerateResponse(
            tikz_code=tokens.markup,
            preview_url=preview_url,
            packages=list(tokens.packages),
            compilation_success=compilation_success,
        )

    @app.post(
        "/api/v1/compile",
        response_model=CompileResponse,
        summary="Compile raw TikZ markup into a raster preview on demand",
    )
    async def compile_tikz(payload: CompileRequest) -> CompileResponse:
        tokens: TikzTokens
        try:
            tokens = TikzTokens(
                markup=payload.tikz_code,
                packages=tuple(payload.packages),
            )
        except DomainError as exc:
            return CompileResponse(
                success=False,
                preview_url=None,
                error=f"Invalid TikZ markup structure: {str(exc)}",
            )

        try:
            compilation_result = await active_compiler.compile_tikz(tokens)
            if compilation_result.is_successful and compilation_result.pdf_data:
                png_bytes = await active_rasterizer.rasterize_pdf(
                    compilation_result.pdf_data
                )
                b64_str = base64.b64encode(png_bytes).decode("ascii")
                return CompileResponse(
                    success=True,
                    preview_url=f"data:image/png;base64,{b64_str}",
                    error=None,
                )
            return CompileResponse(
                success=False,
                preview_url=None,
                error="LaTeX compilation failed. Please verify syntax and packages.",
            )
        except Exception as exc:
            return CompileResponse(
                success=False,
                preview_url=None,
                error=str(exc),
            )

    return app


# Default ASGI entrypoint
app: FastAPI = create_app()
