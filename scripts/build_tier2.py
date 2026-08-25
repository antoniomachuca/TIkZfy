"""Build and persist the Tier 2 compositional corpus (Paso 2).

Generates the compositional SCFG samples, renders them under bounded
concurrency, splits them deterministically, and persists ``(image, markup)``
pairs plus a manifest describing strata, compilation rate, and fingerprints.

References:
    Chomsky, Three Models for the Description of Language — the compositional
        grammar behind the generated corpus.
    Goodfellow et al., Deep Learning — bounded-parallelism mini-batch pipelines.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ensure the parent directory is in the PYTHONPATH so module resolution works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dataset import (
    generate_compositional_batch,
    markup_fingerprint,
    stratified_train_val_split,
)
from core.dataset.packages import BASE_TIKZ_LIBRARIES
from scripts.build_dataset import persist_split, render_corpus

# Tier 1 strata occupy indices 0..7 and the external stratum index 8, so the
# compositional corpus receives stratum index 9.
COMPOSITIONAL_STRATUM: int = 9


async def orchestrate_tier2_build(args: argparse.Namespace) -> None:
    """Generate, render, split, and persist the Tier 2 compositional corpus."""
    started_at: float = time.time()

    print(f"[*] Generating {args.count} compositional samples (seed={args.seed})...")
    markups: list[str] = generate_compositional_batch(args.count, args.seed)

    labels: NDArray[Any] = np.full(len(markups), COMPOSITIONAL_STRATUM, dtype=np.int64)

    print(f"[*] Rendering {len(markups)} samples with {args.workers} workers...")
    payloads, success_mask = await render_corpus(
        markups, args.workers, tikz_libraries=BASE_TIKZ_LIBRARIES
    )

    kept_markups: list[str] = [
        markup for markup, keep in zip(markups, success_mask, strict=True) if keep
    ]
    kept_payloads: list[bytes] = [
        payload for payload, keep in zip(payloads, success_mask, strict=True) if keep
    ]
    kept_labels: NDArray[Any] = labels[success_mask]

    compilation_rate: float = len(kept_markups) / len(markups) if markups else 0.0
    print(
        f"[*] Rendered {len(kept_markups)}/{len(markups)} samples "
        f"(compilation rate {compilation_rate:.2%})."
    )

    train_idx, val_idx = stratified_train_val_split(kept_labels, args.val_ratio, args.seed)

    output_dir: str = args.output_dir
    persist_split(output_dir, "train", train_idx, kept_markups, kept_payloads)
    persist_split(output_dir, "val", val_idx, kept_markups, kept_payloads)

    manifest: dict[str, Any] = {
        "tier": 2,
        "stratum": COMPOSITIONAL_STRATUM,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "requested": args.count,
        "rendered": {
            "total": len(kept_markups),
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "compilation_rate": compilation_rate,
        },
        "tikz_libraries": list(BASE_TIKZ_LIBRARIES),
        "duration_seconds": round(time.time() - started_at, 2),
        "fingerprints_sample": [markup_fingerprint(markup) for markup in kept_markups[:8]],
    }

    manifest_path: str = args.manifest_path
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(
        f"[*] Tier 2 corpus persisted: {manifest['rendered']['train']} train / "
        f"{manifest['rendered']['val']} val pairs. Manifest at '{manifest_path}'."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for the Tier 2 corpus builder."""
    repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Deterministic Tier 2 compositional corpus builder for Image-to-TikZ"
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "dataset", "processed_tier2"),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join(repo_root, "dataset", "manifest_tier2.json"),
    )
    return parser


async def main() -> None:
    cli_args: argparse.Namespace = build_argument_parser().parse_args()
    await orchestrate_tier2_build(cli_args)


if __name__ == "__main__":
    asyncio.run(main())
