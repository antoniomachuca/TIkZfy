import asyncio
import os
import tempfile

from core.exceptions import DomainError
from ports.outbound import ImageRasterizerPort


class GhostscriptRasterizer(ImageRasterizerPort):
    """
    Asynchronous infrastructural adapter for PDF rasterization.

    Implements the ImageRasterizerPort contract by delegating the PDF-to-PNG
    transformation to the Ghostscript subprocess, keeping all OS-level I/O
    outside the mathematical domain.
    """

    def __init__(self, executable: str = "gs") -> None:
        """
        Initializes the Ghostscript rasterization adapter.

        Args:
            executable (str): The Ghostscript binary identifier to invoke.
        """
        self.executable: str = executable

    async def rasterize_pdf(self, pdf_data: bytes, dpi: int = 150) -> bytes:
        """
        Rasterizes a binary PDF payload into PNG-encoded bytes.

        Args:
            pdf_data (bytes): The raw PDF binary payload.
            dpi (int): Output rasterization density in dots per inch.

        Returns:
            bytes: The PNG-encoded raster image payload.

        Raises:
            DomainError: If the subprocess fails or produces no artifact.
        """
        if dpi <= 0:
            raise DomainError(f"Rasterization dpi must be positive. Got {dpi}.")

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path: str = os.path.join(temp_dir, "input.pdf")
            png_path: str = os.path.join(temp_dir, "output.png")

            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_data)

            process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
                self.executable,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-dQUIET",
                "-sDEVICE=png16m",
                f"-r{dpi}",
                f"-sOutputFile={png_path}",
                pdf_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir,
            )

            _, stderr_data = await process.communicate()

            if process.returncode != 0 or not os.path.exists(png_path):
                error_context: str = stderr_data.decode("utf-8", errors="replace")
                raise DomainError(
                    f"Ghostscript rasterization failed. Process output:\n{error_context}"
                )

            with open(png_path, "rb") as png_file:
                png_data: bytes = png_file.read()

            return png_data
