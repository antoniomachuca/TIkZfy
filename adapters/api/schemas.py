"""Pydantic data transfer objects (DTOs) for the FastAPI HTTP adapter."""

from pydantic import BaseModel, Field

from ports.inbound import ImageToTikzUseCase
from ports.outbound import TexCompilerPort

__all__ = [
    "CompileRequest",
    "CompileResponse",
    "GenerateResponse",
    "HealthResponse",
    "ImageToTikzUseCase",
    "TexCompilerPort",
]


class GenerateResponse(BaseModel):
    """Response schema for the TikZ generation endpoint."""

    tikz_code: str = Field(..., description="Generated TikZ/LaTeX markup sequence.")
    preview_url: str | None = Field(
        default=None,
        description="Base64 Data URI preview image of the compiled figure (PNG).",
    )
    packages: list[str] = Field(
        default_factory=list,
        description="Inferred or declared LaTeX package dependencies.",
    )
    compilation_success: bool = Field(
        default=True,
        description="Indicates whether LaTeX compilation and rasterization succeeded.",
    )


class CompileRequest(BaseModel):
    """Request payload for on-demand LaTeX compilation."""

    tikz_code: str = Field(..., min_length=1, description="TikZ markup text to compile.")
    packages: list[str] = Field(
        default_factory=list,
        description="Optional LaTeX packages required by the markup.",
    )


class CompileResponse(BaseModel):
    """Response payload for on-demand LaTeX compilation."""

    success: bool = Field(..., description="Whether TeX Live compiled successfully.")
    preview_url: str | None = Field(
        default=None,
        description="Base64 Data URI of the rasterized PNG preview.",
    )
    error: str | None = Field(
        default=None,
        description="Compilation error message if compilation failed.",
    )


class HealthResponse(BaseModel):
    """Health check response containing system and device telemetry."""

    status: str = Field(default="ok", description="Service health state.")
    device: str = Field(default="cpu", description="Computation device in use.")
    model_loaded: bool = Field(
        default=True,
        description="Whether the neural inference orchestrator is loaded and operational.",
    )
