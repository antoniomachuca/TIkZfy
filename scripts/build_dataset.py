import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

import numpy as np

# Ensure the parent directory is in the PYTHONPATH so module resolution works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.dataset import (
    FAMILY_NAMES,
    deduplicate_markups,
    generate_batch,
    markup_fingerprint,
    stratified_train_val_split,
    within_length_budget,
)
from core.exceptions import DomainError
from core.models import RawLatexDocument, TikzTokens
from core.parser import extract_tikz_graphs

EXTERNAL_STRATUM: int = len(FAMILY_NAMES)


def generate_procedural_corpus(
    per_family_count: int, seed: int
) -> tuple[list[str], np.ndarray]:
    """
    Draws a balanced procedural corpus across every template family.

    Returns:
        tuple[list[str], np.ndarray]: Markups and their stratum labels.
        Shape of labels: (8 * per_family_count,)
    """
    batches: list[list[str]] = [
        generate_batch(family, per_family_count, seed + family_idx)
        for family_idx, family in enumerate(FAMILY_NAMES)
    ]
    markups: list[str] = [markup for batch in batches for markup in batch]
    # Shape: (8 * per_family_count,)
    labels: np.ndarray = np.repeat(np.arange(len(FAMILY_NAMES)), per_family_count)
    return markups, labels


def collect_external_candidates(external_dirs: list[str], cap: int) -> list[str]:
    """
    Scans local .tex sources and extracts tikzpicture blocks within budget.

    Candidates are deduplicated and capped at `cap` entries to bound the
    compilation cost of sources with unknown preamble requirements.
    """
    tex_paths: list[str] = [
        os.path.join(root, name)
        for directory in external_dirs
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(".tex")
    ]

    extracted: list[TikzTokens] = [
        block
        for path in tex_paths
        for block in _safe_extract(path)
    ]
    budgeted: list[str] = [
        tokens.markup for tokens in extracted if within_length_budget(tokens.markup)
    ]
    return deduplicate_markups(budgeted)[:cap]


def _safe_extract(path: str) -> list[TikzTokens]:
    """
    Extracts tikzpicture blocks from one file, absorbing malformed sources.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as source_file:
            return extract_tikz_graphs(RawLatexDocument(raw_text=source_file.read()))
    except (OSError, DomainError):
        return []


async def render_sample(
    semaphore: asyncio.Semaphore,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    tokens: TikzTokens,
) -> bytes | None:
    """
    Compiles and rasterizes one sample under bounded concurrency.

    Returns:
        bytes | None: PNG payload on success, None on any domain failure.
    """
    async with semaphore:
        try:
            compilation = await compiler.compile_tikz(tokens)
            return await rasterizer.rasterize_pdf(compilation.pdf_data)
        except DomainError:
            return None


async def render_corpus(
    markups: list[str], workers: int
) -> tuple[list[bytes], np.ndarray]:
    """
    Renders every markup with bounded parallelism.

    Returns:
        tuple[list[bytes], np.ndarray]: PNG payloads aligned with `markups`
        and a boolean success mask. Shape of mask: (len(markups),)
    """
    semaphore: asyncio.Semaphore = asyncio.Semaphore(workers)
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()

    tasks: list[Any] = [
        render_sample(semaphore, compiler, rasterizer, TikzTokens(markup=markup))
        for markup in markups
    ]
    results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)

    # Shape: (len(markups),)
    success_mask: np.ndarray = np.array(
        [isinstance(result, bytes) for result in results], dtype=bool
    )
    payloads: list[bytes] = [
        result if isinstance(result, bytes) else b"" for result in results
    ]
    return payloads, success_mask


def persist_split(
    output_dir: str,
    split_name: str,
    indices: np.ndarray,
    markups: list[str],
    payloads: list[bytes],
) -> None:
    """
    Persists (markup, image) pairs for one split partition.
    """
    split_dir: str = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for position, sample_idx in enumerate(indices):
        base_path: str = os.path.join(split_dir, f"sample_{position:05d}")
        with open(f"{base_path}.tex", "w", encoding="utf-8") as tex_file:
            tex_file.write(markups[int(sample_idx)])
        with open(f"{base_path}.png", "wb") as png_file:
            png_file.write(payloads[int(sample_idx)])


def extract_showcase_markup(source_path: str) -> str:
    """
    Reassembles the showcase figure: color/macro definitions plus the
    tikzpicture block, so the standalone wrapper compiles it self-contained.
    """
    with open(source_path, encoding="utf-8") as source_file:
        raw_text: str = source_file.read()

    definitions: list[str] = re.findall(
        r"\\definecolor\{[^}]*\}\{[^}]*\}\{[^}]*\}", raw_text
    ) + re.findall(r"\\def\s*\\globalscale\s*\{[^}]*\}", raw_text)

    blocks: list[TikzTokens] = extract_tikz_graphs(
        RawLatexDocument(raw_text=raw_text)
    )
    if not blocks:
        raise DomainError(f"No tikzpicture block found in showcase '{source_path}'.")

    return "\n".join(definitions) + "\n" + blocks[0].markup


async def render_showcase(
    source_path: str | None, showcase_dir: str, workers: int
) -> bool:
    """
    Compiles and rasterizes the showcase figure outside train/val.
    """
    if source_path is None or not os.path.exists(source_path):
        print("[!] Showcase source unavailable; skipping showcase render.")
        return False

    try:
        markup: str = extract_showcase_markup(source_path)
    except DomainError as error:
        print(f"[!] Showcase extraction failed: {error}")
        return False

    semaphore: asyncio.Semaphore = asyncio.Semaphore(workers)
    payload: bytes | None = await render_sample(
        semaphore, AsyncTexLiveAdapter(engine="pdflatex"), GhostscriptRasterizer(),
        TikzTokens(markup=markup),
    )
    if payload is None:
        print("[!] Showcase compilation failed; skipping showcase artifacts.")
        return False

    os.makedirs(showcase_dir, exist_ok=True)
    with open(os.path.join(showcase_dir, "shinji.tex"), "w", encoding="utf-8") as tex_file:
        tex_file.write(markup)
    with open(os.path.join(showcase_dir, "shinji.png"), "wb") as png_file:
        png_file.write(payload)
    return True


async def orchestrate_dataset_build(args: argparse.Namespace) -> None:
    """
    Orchestrates procedural generation, external curation, bounded rendering,
    stratified splitting, and persistence of the (image, markup) dataset.
    """
    started_at: float = time.time()

    print(f"[*] Generating {args.procedural_count} procedural samples...")
    per_family: int = args.procedural_count // len(FAMILY_NAMES)
    markups, labels = generate_procedural_corpus(per_family, args.seed)

    print(f"[*] Collecting up to {args.external_count * 3} external candidates...")
    external_candidates: list[str] = collect_external_candidates(
        args.external_dirs, cap=args.external_count * 3
    )
    print(f"[*] {len(external_candidates)} unique external candidates in budget.")

    external_labels: np.ndarray = np.full(
        len(external_candidates), EXTERNAL_STRATUM, dtype=np.int64
    )
    all_markups: list[str] = markups + external_candidates
    # Shape: (n_total,)
    all_labels: np.ndarray = np.concatenate((labels, external_labels))

    print(f"[*] Rendering {len(all_markups)} samples with {args.workers} workers...")
    payloads, success_mask = await render_corpus(all_markups, args.workers)

    # Keep only successful renders; strata stay aligned via boolean indexing
    kept_markups: list[str] = [
        markup for markup, keep in zip(all_markups, success_mask, strict=True) if keep
    ]
    kept_payloads: list[bytes] = [
        payload for payload, keep in zip(payloads, success_mask, strict=True) if keep
    ]
    kept_labels: np.ndarray = all_labels[success_mask]

    external_success: int = int(
        np.sum(kept_labels == EXTERNAL_STRATUM)
    )
    print(
        f"[*] Rendered {len(kept_markups)}/{len(all_markups)} samples "
        f"({external_success} external successes)."
    )

    train_idx, val_idx = stratified_train_val_split(
        kept_labels, args.val_ratio, args.seed
    )

    output_dir: str = args.output_dir
    persist_split(output_dir, "train", train_idx, kept_markups, kept_payloads)
    persist_split(output_dir, "val", val_idx, kept_markups, kept_payloads)

    print("[*] Rendering showcase figure...")
    showcase_ok: bool = await render_showcase(
        args.showcase_source, args.showcase_dir, args.workers
    )

    # Shape: (n_strata + 1,) per split
    train_counts: np.ndarray = np.bincount(
        kept_labels[train_idx], minlength=EXTERNAL_STRATUM + 1
    )
    val_counts: np.ndarray = np.bincount(
        kept_labels[val_idx], minlength=EXTERNAL_STRATUM + 1
    )

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "requested": {
            "procedural": args.procedural_count,
            "external": args.external_count,
        },
        "rendered": {
            "total": len(kept_markups),
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
        },
        "strata": {
            **{
                family: {
                    "train": int(train_counts[idx]),
                    "val": int(val_counts[idx]),
                }
                for idx, family in enumerate(FAMILY_NAMES)
            },
            "external": {
                "train": int(train_counts[EXTERNAL_STRATUM]),
                "val": int(val_counts[EXTERNAL_STRATUM]),
            },
        },
        "showcase_rendered": showcase_ok,
        "duration_seconds": round(time.time() - started_at, 2),
        "fingerprints_sample": [
            markup_fingerprint(markup) for markup in kept_markups[:8]
        ],
    }

    manifest_path: str = args.manifest_path
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(
        f"[*] Dataset persisted: {manifest['rendered']['train']} train / "
        f"{manifest['rendered']['val']} val pairs. Manifest at '{manifest_path}'."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Builds the CLI contract for the dataset construction entrypoint.
    """
    repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Deterministic (image, markup) dataset builder for Image-to-TikZ"
    )
    parser.add_argument("--procedural-count", type=int, default=4800)
    parser.add_argument("--external-count", type=int, default=200)
    parser.add_argument(
        "--external-dirs",
        nargs="*",
        default=["/usr/local/texlive/2025/texmf-dist/doc/generic/pgf"],
        help="Local directories scanned for real-world TikZ sources.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "dataset", "processed"),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join(repo_root, "dataset", "manifest.json"),
    )
    parser.add_argument(
        "--showcase-source",
        type=str,
        default=os.path.join(repo_root, "..", "shinji.tex"),
    )
    parser.add_argument(
        "--showcase-dir",
        type=str,
        default=os.path.join(repo_root, "dataset", "showcase"),
    )
    return parser


async def main():
    cli_args: argparse.Namespace = build_argument_parser().parse_args()
    await orchestrate_dataset_build(cli_args)


if __name__ == "__main__":
    asyncio.run(main())
