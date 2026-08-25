"""Ingest and filter the DaTikZ-V4 dataset into a Tier 3 OOD test corpus (Paso 3).

Downloads the public ``nllg/DaTikZ-V4`` dataset, filters rows by decoder budget
and presence of a ``tikzpicture`` block, extracts the drawing blocks, deduplicates
them, and compiles/rasterizes the survivors into ``dataset/processed_tier3/test``.

The result is an out-of-distribution evaluation set of real scientific figures —
it is never used for training.

References:
    NLLG Lab, DaTikZ-V4 — large-scale TikZ corpus from arXiv, GitHub and TeXample.
    Evans, Domain-Driven Design — external data ingested through an adapter at
        the port boundary, never leaking into the pure domain.
"""

import argparse
import asyncio
import io
import json
import os
import ssl
import sys
import time
import urllib.request
from collections.abc import Iterator
from typing import Any

# Ensure the parent directory is in the PYTHONPATH so module resolution works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.dataset import markup_fingerprint, within_length_budget
from core.dataset.packages import BASE_TIKZ_LIBRARIES, detect_required_packages
from core.exceptions import DomainError
from core.models import RawLatexDocument, TikzTokens
from core.parser import extract_tikz_graphs

SHARD_URL_TEMPLATE: str = (
    "https://huggingface.co/datasets/nllg/DaTikZ-V4/resolve/main/data/train-{index:05d}.parquet"
)
SHARD_COUNT: int = 86


def _ssl_context() -> ssl.SSLContext:
    """Return a TLS context that trusts the certifi CA bundle (macOS Python fix)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with the datasets dependency
        return ssl.create_default_context()


def iter_datikz_rows(
    max_rows: int, shard_start: int = 0, shard_count: int = SHARD_COUNT
) -> Iterator[dict[str, Any]]:
    """Yield DaTikZ-V4 row dicts by downloading and reading parquet shards.

    The ``datasets`` library's streaming loader can stall on this corpus, so the
    shards are fetched directly and read with ``pyarrow`` (a transitive dependency
    of ``datasets``). Each row's ``tikz_code`` column holds a full LaTeX document.
    """
    import pyarrow.parquet as pq

    context: ssl.SSLContext = _ssl_context()
    emitted: int = 0
    for shard in range(shard_start, shard_count):
        url: str = SHARD_URL_TEMPLATE.format(index=shard)
        print(f"[*] Downloading shard {shard} ({shard + 1}/{shard_count})...")
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response:
                payload: bytes = response.read()
        except OSError as error:
            print(f"[!] Shard {shard} download failed: {error}")
            continue

        table = pq.read_table(io.BytesIO(payload))
        for row in table.to_pylist():
            yield row
            emitted += 1
            if emitted >= max_rows:
                return


def extract_markup_from_code(code: str) -> str | None:
    """Extract the first tikzpicture block from a DaTikZ ``tikz_code`` field."""
    blocks: list[TikzTokens] = extract_tikz_graphs(RawLatexDocument(raw_text=code))
    if not blocks:
        return None
    return blocks[0].markup


def filter_candidates(rows: Any, max_rows: int, candidate_cap: int) -> list[tuple[str, str]]:
    """Stream DaTikZ rows into deduplicated ``(block, full_code)`` candidates.

    The DaTikZ-V4 schema stores the drawing source in the ``tikz_code`` column.
    The extracted ``tikzpicture`` block becomes the reference markup, while the
    full document is kept for faithful ground-truth rendering.
    """
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for position, row in enumerate(rows):
        if position >= max_rows:
            break
        if len(candidates) >= candidate_cap:
            break
        code: str = row.get("tikz_code", "")
        if not code or "\\begin{tikzpicture}" not in code:
            continue
        if not within_length_budget(code, max_chars=4000):
            continue
        markup: str | None = extract_markup_from_code(code)
        if markup is None or not within_length_budget(markup, max_chars=4000):
            continue
        if markup in seen:
            continue
        seen.add(markup)
        candidates.append((markup, code))
    return deduplicate_markups_candidates(candidates)


def deduplicate_markups_candidates(
    candidates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Deduplicate candidates by their extracted markup, preserving first order."""
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for markup, code in candidates:
        if markup in seen:
            continue
        seen.add(markup)
        unique.append((markup, code))
    return unique


async def compile_one(
    semaphore: asyncio.Semaphore,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    full_code: str,
    timeout: float,
) -> bytes | None:
    """Compile one full document and return its rasterized PNG on success."""
    async with semaphore:
        try:
            tokens: TikzTokens = TikzTokens(
                markup=full_code, packages=detect_required_packages(full_code)
            )
            compilation = await asyncio.wait_for(compiler.compile_tikz(tokens), timeout=timeout)
            return await asyncio.wait_for(
                rasterizer.rasterize_pdf(compilation.pdf_data), timeout=timeout
            )
        except (DomainError, asyncio.TimeoutError):
            return None


async def compile_candidates(
    candidates: list[tuple[str, str]], workers: int, target: int, timeout: float
) -> tuple[list[str], list[bytes]]:
    """Compile candidates under bounded concurrency, keeping up to ``target``."""
    semaphore: asyncio.Semaphore = asyncio.Semaphore(workers)
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(
        engine="pdflatex", tikz_libraries=BASE_TIKZ_LIBRARIES
    )
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()

    kept_markups: list[str] = []
    kept_payloads: list[bytes] = []

    for start in range(0, len(candidates), workers):
        if len(kept_markups) >= target:
            break
        chunk: list[tuple[str, str]] = candidates[start : start + workers]
        results = await asyncio.gather(
            *[
                compile_one(semaphore, compiler, rasterizer, full_code, timeout)
                for _, full_code in chunk
            ],
            return_exceptions=True,
        )
        for (markup, _), result in zip(chunk, results, strict=True):
            if isinstance(result, bytes):
                kept_markups.append(markup)
                kept_payloads.append(result)
            if len(kept_markups) >= target:
                break

    return kept_markups, kept_payloads


def persist_test_set(output_dir: str, markups: list[str], payloads: list[bytes]) -> None:
    """Persist the compiled Tier 3 test samples as ``(image, markup)`` pairs."""
    test_dir: str = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    for position, (markup, payload) in enumerate(zip(markups, payloads, strict=True)):
        base_path: str = os.path.join(test_dir, f"sample_{position:05d}")
        with open(f"{base_path}.tex", "w", encoding="utf-8") as tex_file:
            tex_file.write(markup)
        with open(f"{base_path}.png", "wb") as png_file:
            png_file.write(payload)


async def orchestrate_tier3_ingestion(args: argparse.Namespace) -> None:
    """Download, filter, compile, and persist the Tier 3 test corpus."""
    started_at: float = time.time()

    print(f"[*] Streaming DaTikZ-V4 parquet shards (cap={args.max_rows})...")
    rows: Iterator[dict[str, Any]] = iter_datikz_rows(
        args.max_rows, args.shard_start, args.shard_count
    )
    candidates: list[tuple[str, str]] = filter_candidates(rows, args.max_rows, args.candidate_cap)
    print(f"[*] {len(candidates)} unique candidates after filtering/dedup.")

    print(f"[*] Compiling candidates (target={args.target}, workers={args.workers})...")
    kept_markups, kept_payloads = await compile_candidates(
        candidates, args.workers, args.target, args.timeout
    )
    print(f"[*] {len(kept_markups)} samples compiled successfully.")

    persist_test_set(args.output_dir, kept_markups, kept_payloads)

    manifest: dict[str, Any] = {
        "tier": 3,
        "dataset": "nllg/DaTikZ-V4",
        "requested_target": args.target,
        "candidates_scanned": len(candidates),
        "compiled": len(kept_markups),
        "tikz_libraries": list(BASE_TIKZ_LIBRARIES),
        "duration_seconds": round(time.time() - started_at, 2),
        "fingerprints_sample": [markup_fingerprint(markup) for markup in kept_markups[:8]],
    }

    manifest_path: str = args.manifest_path
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"[*] Tier 3 corpus persisted. Manifest at '{manifest_path}'.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for Tier 3 ingestion."""
    repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Ingest and filter DaTikZ-V4 into a Tier 3 OOD test corpus."
    )
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--candidate-cap", type=int, default=5000)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "dataset", "processed_tier3"),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join(repo_root, "dataset", "manifest_tier3.json"),
    )
    return parser


async def main() -> None:
    cli_args: argparse.Namespace = build_argument_parser().parse_args()
    await orchestrate_tier3_ingestion(cli_args)


if __name__ == "__main__":
    asyncio.run(main())
