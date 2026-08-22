import asyncio
import os
import re
import tempfile

from core.dataset.packages import build_preamble
from core.exceptions import CompilationSyntaxError, DomainError, MissingPackageError
from core.models import CompilationResult, TikzTokens
from ports.outbound import TexCompilerPort

# TeX Live emits ``! LaTeX Error: File `pgfplots.sty' not found.`` when a
# ``\\usepackage`` target is absent from the installation.
_MISSING_PACKAGE_PATTERN: re.Pattern[str] = re.compile(r"File `([^`']+)' not found")


def categorize_compilation_failure(log_text: str) -> DomainError:
    """
    Categorizes a failed TeX compilation from its engine log.

    Distinguishes a missing package (``File `X.sty' not found``) from an
    invalid-syntax failure, returning the failing package name on the former.

    Args:
        log_text (str): Decoded TeX Live engine output from a failed compile.

    Returns:
        DomainError: A ``MissingPackageError`` when the log reports an absent
            package, otherwise a ``CompilationSyntaxError``.

    Temporal complexity: O(L) where L is the log length (single regex scan).
    """
    missing_match: re.Match[str] | None = _MISSING_PACKAGE_PATTERN.search(log_text)
    if missing_match is not None:
        missing_package: str = missing_match.group(1)
        if missing_package.endswith(".sty"):
            missing_package = missing_package[: -len(".sty")]
        return MissingPackageError(
            f"Required LaTeX package '{missing_package}' is not installed. "
            "Engine output:\n" + log_text
        )
    return CompilationSyntaxError(
        "TeX compilation structurally failed. Engine output:\n" + log_text
    )


class AsyncTexLiveAdapter(TexCompilerPort):
    """
    Asynchronous infrastructural adapter for TeX Live compilation.

    Implements the TexCompilerPort interface to orchestrate the generation
    of PDF binary artifacts via OS-level subprocesses without blocking the
    primary application thread. The standalone preamble is resolved from the
    package catalog declared on ``TikzTokens.packages``, and compilation
    failures are categorized as either a missing-package or a syntax error.
    """

    def __init__(self, engine: str = "pdflatex", tikz_libraries: tuple[str, ...] = ()) -> None:
        """
        Initializes the TeX Live adapter.

        Args:
            engine (str): The compiler engine to invoke (e.g., 'pdflatex', 'lualatex').
            tikz_libraries (tuple[str, ...]): Optional TikZ library names injected
                as usetikzlibrary lines in the standalone wrapper. Empty by default,
                preserving the minimal production compilation preamble.
        """
        self.engine: str = engine
        self.tikz_libraries: tuple[str, ...] = tikz_libraries

    async def compile_tikz(self, tokens: TikzTokens) -> CompilationResult:
        """
        Compiles the bounded syntactic sequence asynchronously.

        Args:
            tokens (TikzTokens): The structurally validated LaTeX sequence.

        Returns:
            CompilationResult: The product of compilation encapsulating success status
                               and arbitrary binary payload data.

        Raises:
            MissingPackageError: If a required package is not installed in TeX Live.
            CompilationSyntaxError: If the markup syntax is structurally invalid.
            DomainError: If subprocess mapping errors occur.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file_path: str = os.path.join(temp_dir, "document.tex")
            markup: str = tokens.markup

            if "\\documentclass" not in markup:
                preamble: str = build_preamble(tokens.packages, self.tikz_libraries)
                markup = (
                    f"{preamble}"
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
                raise categorize_compilation_failure(error_context)

            return CompilationResult(pdf_data=pdf_data, is_successful=is_successful)
