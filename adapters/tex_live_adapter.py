import asyncio
import os
import tempfile

from core.exceptions import DomainError
from core.models import CompilationResult, TikzTokens
from ports.outbound import TexCompilerPort


class AsyncTexLiveAdapter(TexCompilerPort):
    """
    Asynchronous infrastructural adapter for TeX Live compilation.

    Implements the TexCompilerPort interface to orchestrate the generation
    of PDF binary artifacts via OS-level subprocesses without blocking the
    primary application thread.
    """

    def __init__(self, engine: str = "pdflatex") -> None:
        """
        Initializes the TeX Live adapter.

        Args:
            engine (str): The compiler engine to invoke (e.g., 'pdflatex', 'lualatex').
        """
        self.engine: str = engine

    async def compile_tikz(self, tokens: TikzTokens) -> CompilationResult:
        """
        Compiles the bounded syntactic sequence asynchronously.

        Args:
            tokens (TikzTokens): The structurally validated LaTeX sequence.

        Returns:
            CompilationResult: The product of compilation encapsulating success status
                               and arbitrary binary payload data.

        Raises:
            DomainError: If the execution strictly fails or subprocess mapping errors occur.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file_path: str = os.path.join(temp_dir, "document.tex")
            markup: str = tokens.markup

            if "\\documentclass" not in markup:
                markup = (
                    "\\documentclass{standalone}\n"
                    "\\usepackage{tikz}\n"
                    "\\begin{document}\n"
                    f"{markup}\n"
                    "\\end{document}\n"
                )

            with open(tex_file_path, "w", encoding="utf-8") as tex_file:
                tex_file.write(markup)

            process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
                self.engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory", temp_dir,
                "document.tex",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )

            stdout_data, stderr_data = await process.communicate()

            is_successful: bool = (process.returncode == 0)
            pdf_data: bytes = b""

            pdf_file_path: str = os.path.join(temp_dir, "document.pdf")

            if is_successful and os.path.exists(pdf_file_path):
                with open(pdf_file_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
            else:
                error_context: str = stdout_data.decode("utf-8", errors="replace")
                raise DomainError(
                    "TeX compilation structurally failed. Engine output:\n"
                    f"{error_context}"
                )

            return CompilationResult(pdf_data=pdf_data, is_successful=is_successful)
