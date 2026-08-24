"""Master Phase 3 massive corpus generation, filtering, and tensor encoding pipeline.

Generates and encodes the full experimental corpus (N = 5,000 samples):
    1. Tier 1 Canonical Procedural: 2,000 samples across 8 geometric template families.
    2. Tier 2 Compositional SCFG: 2,000 multi-package hierarchical samples.
    3. Tier 3 DaTikZ-V4 In-The-Wild: 1,000 filtered out-of-distribution test samples.
    4. Encodes and persists standardized PyTorch tensors under ``dataset/encoded/``:
       - ``train_images.pt``, ``train_tokens.pt`` (Merged Tier 1 + Tier 2 Train)
       - ``val_images.pt``, ``val_tokens.pt`` (Tier 1 Validation)
       - ``tier2_val_images.pt``, ``tier2_val_tokens.pt`` (Tier 2 Validation)
       - ``tier3_test_images.pt``, ``tier3_test_tokens.pt`` (Tier 3 OOD Evaluation Set)
       - ``vocabulary.json`` (Coordinate-quantized shared vocabulary)

References:
    Goodfellow et al., Deep Learning — data diversification and batch tensor ingestion.
    Chomsky, Three Models for the Description of Language — generative grammars.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.torchvision_loader import TorchVisionImageLoader
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset import (
    FAMILY_NAMES,
    generate_batch,
    generate_compositional_batch,
    stratified_train_val_split,
)
from core.dataset.packages import BASE_TIKZ_LIBRARIES
from core.math.spatial import normalize_channels, resize_spatial_dimensions
from core.math.tokenization import batch_encode, build_vocabulary
from core.models import ImageTensor, TikzTokens, TokenVocabulary
from scripts.build_dataset import persist_split, render_corpus
from scripts.ingest_datikz import (
    compile_candidates,
    filter_candidates,
    iter_datikz_rows,
    persist_test_set,
)


def _coerce_to_three_channels(image: torch.Tensor) -> torch.Tensor:
    """Map an ``(N, C, H, W)`` image tensor onto exactly 3 RGB channels."""
    if image.shape[1] == 3:
        return image
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    if image.shape[1] == 4:
        return image[:, :3, :, :]
    raise ValueError(f"Unsupported channel count {image.shape[1]}.")


def load_split_image_tensor(
    split_dir: Path, target_height: int, target_width: int
) -> torch.Tensor:
    """Load, normalize, and resize all PNG images in split_dir into (N, 3, H, W)."""
    image_paths: list[Path] = sorted(split_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG files found in '{split_dir}'.")

    loader: TorchVisionImageLoader = TorchVisionImageLoader()
    resized_tensors: list[torch.Tensor] = []
    for img_path in image_paths:
        rgb_tensor: torch.Tensor = _coerce_to_three_channels(
            loader.load_image(str(img_path)).raw_tensor
        )
        norm_img: ImageTensor = normalize_channels(ImageTensor(rgb_tensor))
        resized: ImageTensor = resize_spatial_dimensions(
            norm_img, target_height, target_width
        )
        resized_tensors.append(resized.raw_tensor)
    return torch.cat(resized_tensors, dim=0)


def load_split_markup_corpus(split_dir: Path) -> list[TikzTokens]:
    """Load all .tex files in split_dir in sorted order."""
    tex_paths: list[Path] = sorted(split_dir.glob("*.tex"))
    if not tex_paths:
        raise ValueError(f"No TEX files found in '{split_dir}'.")
    return [TikzTokens(markup=p.read_text(encoding="utf-8")) for p in tex_paths]


async def generate_tier1_dataset(
    target_count: int,
    workers: int,
    output_dir: Path,
    seed: int = 42,
    val_ratio: float = 0.1,
) -> None:
    """Generate and render balanced Tier 1 procedural corpus."""
    print(f"\n[*] === Generating Tier 1 Procedural Dataset ({target_count} target) ===")
    per_family: int = max(1, target_count // len(FAMILY_NAMES))
    batches: list[list[str]] = [
        generate_batch(fam, per_family, seed + idx)
        for idx, fam in enumerate(FAMILY_NAMES)
    ]
    all_markups: list[str] = [m for b in batches for m in b]
    labels: NDArray[Any] = np.repeat(np.arange(len(FAMILY_NAMES)), per_family)

    print(f"[*] Rendering {len(all_markups)} Tier 1 markups with {workers} workers...")
    payloads, success_mask = await render_corpus(all_markups, workers)

    kept_markups: list[str] = [
        m for m, keep in zip(all_markups, success_mask, strict=True) if keep
    ]
    kept_payloads: list[bytes] = [
        p for p, keep in zip(payloads, success_mask, strict=True) if keep
    ]
    kept_labels: NDArray[Any] = labels[success_mask]

    print(f"[*] Compiled {len(kept_markups)}/{len(all_markups)} Tier 1 samples.")
    train_idx, val_idx = stratified_train_val_split(kept_labels, val_ratio, seed)

    persist_split(str(output_dir), "train", train_idx, kept_markups, kept_payloads)
    persist_split(str(output_dir), "val", val_idx, kept_markups, kept_payloads)
    print(f"[+] Tier 1 saved to '{output_dir}'.")


async def generate_tier2_dataset(
    target_count: int,
    workers: int,
    output_dir: Path,
    seed: int = 42,
    val_ratio: float = 0.1,
) -> None:
    """Generate and render Tier 2 compositional SCFG corpus."""
    print(f"\n[*] === Generating Tier 2 Compositional Dataset ({target_count} target) ===")
    markups: list[str] = generate_compositional_batch(target_count, seed)
    labels: NDArray[Any] = np.full(len(markups), 9, dtype=np.int64)

    print(f"[*] Rendering {len(markups)} Tier 2 markups with {workers} workers...")
    payloads, success_mask = await render_corpus(
        markups, workers, tikz_libraries=BASE_TIKZ_LIBRARIES
    )

    kept_markups: list[str] = [
        m for m, keep in zip(markups, success_mask, strict=True) if keep
    ]
    kept_payloads: list[bytes] = [
        p for p, keep in zip(payloads, success_mask, strict=True) if keep
    ]
    kept_labels: NDArray[Any] = labels[success_mask]

    print(f"[*] Compiled {len(kept_markups)}/{len(markups)} Tier 2 samples.")
    train_idx, val_idx = stratified_train_val_split(kept_labels, val_ratio, seed)

    persist_split(str(output_dir), "train", train_idx, kept_markups, kept_payloads)
    persist_split(str(output_dir), "val", val_idx, kept_markups, kept_payloads)
    print(f"[+] Tier 2 saved to '{output_dir}'.")


async def generate_tier3_dataset(
    target_count: int,
    workers: int,
    output_dir: Path,
    max_scan_rows: int = 15000,
) -> None:
    """Download, filter, and render DaTikZ-V4 samples for Tier 3 OOD test."""
    print(f"\n[*] === Ingesting Tier 3 DaTikZ-V4 Dataset ({target_count} target) ===")
    rows_stream = iter_datikz_rows(max_scan_rows)
    candidates: list[tuple[str, str]] = filter_candidates(
        rows_stream, max_scan_rows, target_count * 2
    )
    print(f"[*] Collected {len(candidates)} candidate markups from DaTikZ shards.")

    print(f"[*] Compiling up to {target_count} Tier 3 samples with {workers} workers...")
    kept_markups, kept_payloads = await compile_candidates(
        candidates, workers=workers, target=target_count, timeout=15.0
    )
    persist_test_set(str(output_dir), kept_markups, kept_payloads)
    print(f"[+] Tier 3 persisted {len(kept_markups)} OOD evaluation pairs in '{output_dir}'.")


def encode_all_corpora(
    tier1_dir: Path,
    tier2_dir: Path,
    tier3_dir: Path,
    encoded_dir: Path,
    max_length: int = 512,
    target_height: int = 64,
    target_width: int = 64,
) -> None:
    """Encode all datasets into standardized PyTorch tensors under encoded_dir."""
    print("\n[*] === Encoding All Datasets to Compact Tensors ===")
    encoded_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save shared vocabulary from Tier 1 + Tier 2 Train
    t1_train_markups: list[TikzTokens] = load_split_markup_corpus(tier1_dir / "train")
    t2_train_markups: list[TikzTokens] = load_split_markup_corpus(tier2_dir / "train")
    combined_train_markups: list[TikzTokens] = t1_train_markups + t2_train_markups

    print(f"[*] Building vocabulary from {len(combined_train_markups)} training markups...")
    vocabulary: TokenVocabulary = build_vocabulary(combined_train_markups)
    vocab_path: Path = encoded_dir / "vocabulary.json"
    JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocab_path))
    print(f"[+] Vocabulary persisted ({len(vocabulary.token_to_index)} tokens) to '{vocab_path}'.")

    # 2. Encode Tier 1 + Tier 2 Merged Training Set
    print("[*] Encoding Merged Training Images & Tokens...")
    t1_train_img: torch.Tensor = load_split_image_tensor(
        tier1_dir / "train", target_height, target_width
    )
    t2_train_img: torch.Tensor = load_split_image_tensor(
        tier2_dir / "train", target_height, target_width
    )
    train_images: torch.Tensor = torch.cat([t1_train_img, t2_train_img], dim=0)
    train_tokens: torch.Tensor = batch_encode(
        combined_train_markups, vocabulary, max_length
    )

    torch.save(train_images, encoded_dir / "train_images.pt")
    torch.save(train_tokens, encoded_dir / "train_tokens.pt")
    print(
        f"[+] Train tensors saved: Images {tuple(train_images.shape)}, "
        f"Tokens {tuple(train_tokens.shape)}"
    )

    # 3. Encode Tier 1 Val
    print("[*] Encoding Tier 1 Validation...")
    t1_val_markups: list[TikzTokens] = load_split_markup_corpus(tier1_dir / "val")
    t1_val_img: torch.Tensor = load_split_image_tensor(
        tier1_dir / "val", target_height, target_width
    )
    t1_val_tok: torch.Tensor = batch_encode(t1_val_markups, vocabulary, max_length)
    torch.save(t1_val_img, encoded_dir / "val_images.pt")
    torch.save(t1_val_tok, encoded_dir / "val_tokens.pt")
    print(
        f"[+] Tier 1 Val tensors saved: Images {tuple(t1_val_img.shape)}, "
        f"Tokens {tuple(t1_val_tok.shape)}"
    )

    # 4. Encode Tier 2 Val
    print("[*] Encoding Tier 2 Validation...")
    t2_val_markups: list[TikzTokens] = load_split_markup_corpus(tier2_dir / "val")
    t2_val_img: torch.Tensor = load_split_image_tensor(
        tier2_dir / "val", target_height, target_width
    )
    t2_val_tok: torch.Tensor = batch_encode(t2_val_markups, vocabulary, max_length)
    torch.save(t2_val_img, encoded_dir / "tier2_val_images.pt")
    torch.save(t2_val_tok, encoded_dir / "tier2_val_tokens.pt")
    print(
        f"[+] Tier 2 Val tensors saved: Images {tuple(t2_val_img.shape)}, "
        f"Tokens {tuple(t2_val_tok.shape)}"
    )

    # 5. Encode Tier 3 Test
    print("[*] Encoding Tier 3 OOD Test...")
    t3_test_dir: Path = tier3_dir / "test"
    t3_test_markups: list[TikzTokens] = load_split_markup_corpus(t3_test_dir)
    t3_test_img: torch.Tensor = load_split_image_tensor(
        t3_test_dir, target_height, target_width
    )
    t3_test_tok: torch.Tensor = batch_encode(t3_test_markups, vocabulary, max_length)
    torch.save(t3_test_img, encoded_dir / "tier3_test_images.pt")
    torch.save(t3_test_tok, encoded_dir / "tier3_test_tokens.pt")
    print(
        f"[+] Tier 3 Test tensors saved: Images {tuple(t3_test_img.shape)}, "
        f"Tokens {tuple(t3_test_tok.shape)}"
    )
    print(f"\n[✓] All 5,000 samples successfully encoded and persisted in '{encoded_dir}'.")


async def main() -> None:
    """CLI orchestrator entrypoint."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Master Phase 3 Massive Corpus Builder & Encoder."
    )
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser.add_argument("--tier1-count", type=int, default=2000)
    parser.add_argument("--tier2-count", type=int, default=2000)
    parser.add_argument("--tier3-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dataset-root", type=Path, default=repo_root / "dataset"
    )
    args: argparse.Namespace = parser.parse_args()

    t1_dir: Path = args.dataset_root / "processed_tier1"
    t2_dir: Path = args.dataset_root / "processed_tier2"
    t3_dir: Path = args.dataset_root / "processed_tier3"
    enc_dir: Path = args.dataset_root / "encoded"

    start_time: float = time.time()
    await generate_tier1_dataset(args.tier1_count, args.workers, t1_dir, args.seed)
    await generate_tier2_dataset(args.tier2_count, args.workers, t2_dir, args.seed)
    await generate_tier3_dataset(args.tier3_count, args.workers, t3_dir)
    encode_all_corpora(t1_dir, t2_dir, t3_dir, enc_dir, max_length=args.max_length)
    print(f"\n[*] Total pipeline execution time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
