import argparse
import asyncio
import os
import sys
from typing import Any

# Ensure the parent directory is in the PYTHONPATH so module resolution works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.latex_ingestion_client import AiohttpLatexClient
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.exceptions import DomainError
from core.models import CompilationResult, RawLatexDocument, TikzTokens


async def orchestrate_data_generation(uris: list[str], output_dir: str) -> None:
    """
    Orchestrates the deterministic fetching, structural validation, and asynchronous
    compilation of spatial datasets in O(1) blocking time per file.

    Args:
        uris (list[str]): The URIs containing raw LaTeX string payloads.
        output_dir (str): The persistent storage directory for the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    ingestion_client: AiohttpLatexClient = AiohttpLatexClient()
    compiler_adapter: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")

    print(f"[*] Ingesting {len(uris)} potential LaTeX sources...")
    raw_docs: list[RawLatexDocument] = await ingestion_client.fetch_sources(uris)

    print(f"[*] Successfully ingested {len(raw_docs)} unparsed documents.")

    tokens_list: list[TikzTokens] = []

    # Deterministic mapping loop
    for doc in raw_docs:
        try:
            tokens: TikzTokens = TikzTokens(markup=doc.raw_text)
            tokens_list.append(tokens)
        except DomainError as e:
            print(f"[!] Topological constraint violation during tokenization: {e}")

    print(f"[*] Compiling {len(tokens_list)} syntactically validated TikZ sequences...")

    compile_tasks: list[Any] = [compiler_adapter.compile_tikz(tokens) for tokens in tokens_list]
    compilation_results: list[Any] = await asyncio.gather(*compile_tasks, return_exceptions=True)

    success_count: int = 0

    for idx, (result, tokens) in enumerate(zip(compilation_results, tokens_list, strict=True)):
        if isinstance(result, CompilationResult) and result.is_successful:
            base_filename: str = os.path.join(output_dir, f"sample_{success_count:04d}")

            with open(f"{base_filename}.tex", "w", encoding="utf-8") as tex_out:
                tex_out.write(tokens.markup)

            with open(f"{base_filename}.pdf", "wb") as pdf_out:
                pdf_out.write(result.pdf_data)

            success_count += 1
        elif isinstance(result, Exception):
            print(f"[!] Compilation strictly failed for item {idx}: {result}")

    print(
        f"[*] Orchestration completed. Persisted {success_count} structural "
        f"pairs in '{output_dir}'."
    )


parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Deterministic Dataset Orchestrator for Image-to-TikZ"
)
parser.add_argument(
    "--uris", nargs="*", default=[], help="List of HTTP URIs to fetch raw TikZ sources from."
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"),
    help="Target directory for the generated PDF and TEX files.",
)

args = parser.parse_args()

asyncio.run(orchestrate_data_generation(args.uris, args.output_dir))
