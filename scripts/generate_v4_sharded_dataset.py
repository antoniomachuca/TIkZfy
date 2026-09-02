"""High-throughput generation script for Dataset V4 with sharded storage.

Produces balanced geometric samples across 8 canonical families at 256x256 resolution,
injects family prefix conditioning tokens (<FAM:xxx>), and persists memory-mapped shards
(10,000 samples each) alongside DatasetV4Manifest descriptors.

References:
    Golub & Van Loan, Matrix Computations - discrete coordinate lattice mapping.
    Goodfellow et al., Deep Learning - dataset synthesis and distributed training I/O.
"""

import argparse
import asyncio
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.sharded import DatasetV4Manifest, ShardMetadata, save_shard
from core.dataset.templates import FAMILY_NAMES, family_index, generate_sample
from core.math.tokenization import build_vocabulary, encode_to_tensor
from core.models import (
    TikzTokens,
    TokenVocabulary,
)


def _build_v4_vocabulary(target_path: Path) -> TokenVocabulary:
    """Construct and persist the canonical V4 vocabulary with family tokens."""
    seed_corpus: list[TikzTokens] = []
    rng: np.random.Generator = np.random.default_rng(12345)

    for fam in FAMILY_NAMES:
        sample_idx = 0
        while sample_idx < 100:
            markup: str = generate_sample(fam, rng)
            seed_corpus.append(TikzTokens(markup=f"<FAM:{fam}> {markup}"))
            sample_idx += 1

    vocab: TokenVocabulary = build_vocabulary(seed_corpus, coordinate_step=0.05)
    JsonVocabularyAdapter().save_vocabulary(vocab, str(target_path))
    return vocab


async def _render_and_encode_sample(
    raw_markup: str,
    family: str,
    vocabulary: TokenVocabulary,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    semaphore: asyncio.Semaphore,
    image_size: int = 256,
    max_length: int = 512,
) -> tuple[bool, torch.Tensor | None, torch.Tensor | None, int]:
    """Compile raw TikZ to PDF, rasterize to 256x256 uint8, and encode prefix tokens."""
    async with semaphore:
        try:
            comp_res = await compiler.compile_tikz(TikzTokens(markup=raw_markup))
            if not comp_res.is_successful or not comp_res.pdf_data:
                return False, None, None, -1

            png_bytes = await rasterizer.rasterize_pdf(comp_res.pdf_data, dpi=144)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            t = torch.from_numpy(arr).permute(2, 0, 1)  # Shape: (3, H, W) in [0, 255]

            if t.shape[1:] != (image_size, image_size):
                # Float interpolation then convert back to uint8
                float_t = t.to(torch.float32).unsqueeze(0)
                resized = F.interpolate(
                    float_t,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                t = resized.clamp(0, 255).to(torch.uint8)

            full_markup = f"<FAM:{family}> {raw_markup}"
            encoded_tokens = encode_to_tensor(
                TikzTokens(markup=full_markup),
                vocabulary,
                max_length=max_length,
                coordinate_step=0.05,
            )
            return True, t, encoded_tokens, family_index(family)
        except Exception:
            return False, None, None, -1


async def generate_split_shards(
    split_name: str,
    output_dir: Path,
    samples_per_family: int,
    shard_size: int,
    vocabulary: TokenVocabulary,
    concurrency: int = 8,
    seed: int = 42,
    image_size: int = 256,
    max_length: int = 512,
) -> list[ShardMetadata]:
    """Generate all shards for a specific dataset split (train, val, or test)."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()
    semaphore = asyncio.Semaphore(concurrency)

    rng = np.random.Generator(np.random.PCG64(seed))

    buffer_images: list[torch.Tensor] = []
    buffer_tokens: list[torch.Tensor] = []
    buffer_labels: list[int] = []
    buffer_family_counts: dict[str, int] = dict.fromkeys(FAMILY_NAMES, 0)

    shards_metadata: list[ShardMetadata] = []
    shard_counter: int = 0

    print(
        f"[*] Starting split '{split_name}': {samples_per_family} samples/family "
        f"x {len(FAMILY_NAMES)} families = {samples_per_family * len(FAMILY_NAMES)} target samples."
    )

    for fam in FAMILY_NAMES:
        collected: int = 0
        attempts: int = 0
        max_attempts: int = samples_per_family * 3

        while collected < samples_per_family and attempts < max_attempts:
            # Batch size for async generation
            batch_size = min(concurrency * 2, samples_per_family - collected)
            raw_markups = [generate_sample(fam, rng) for _ in range(batch_size)]
            attempts += batch_size

            tasks = [
                _render_and_encode_sample(
                    mk,
                    fam,
                    vocabulary,
                    compiler,
                    rasterizer,
                    semaphore,
                    image_size=image_size,
                    max_length=max_length,
                )
                for mk in raw_markups
            ]
            results = await asyncio.gather(*tasks)

            res_idx = 0
            while res_idx < len(results) and collected < samples_per_family:
                ok, img_tensor, tok_tensor, fam_idx = results[res_idx]
                if ok and img_tensor is not None and tok_tensor is not None:
                    buffer_images.append(img_tensor)
                    buffer_tokens.append(tok_tensor)
                    buffer_labels.append(fam_idx)
                    buffer_family_counts[fam] += 1
                    collected += 1

                    # Check if active shard is full
                    if len(buffer_images) == shard_size:
                        shard_filename = f"{split_name}_shard_{shard_counter:04d}.pt"
                        shard_path = split_dir / shard_filename
                        save_shard(
                            shard_path,
                            torch.stack(buffer_images),
                            torch.stack(buffer_tokens),
                            torch.tensor(buffer_labels, dtype=torch.long),
                        )
                        meta = ShardMetadata(
                            shard_id=shard_counter,
                            file_name=f"{split_name}/{shard_filename}",
                            num_samples=len(buffer_images),
                            family_counts=dict(buffer_family_counts),
                        )
                        shards_metadata.append(meta)
                        print(f"  [+] Persisted {shard_filename} ({len(buffer_images)} samples).")
                        shard_counter += 1
                        buffer_images.clear()
                        buffer_tokens.clear()
                        buffer_labels.clear()
                        buffer_family_counts = dict.fromkeys(FAMILY_NAMES, 0)

                res_idx += 1

        print(f"  [+] Family '{fam}' completed: {collected}/{samples_per_family} valid samples.")

    # Flush any remaining buffer into a final shard
    if buffer_images:
        shard_filename = f"{split_name}_shard_{shard_counter:04d}.pt"
        shard_path = split_dir / shard_filename
        save_shard(
            shard_path,
            torch.stack(buffer_images),
            torch.stack(buffer_tokens),
            torch.tensor(buffer_labels, dtype=torch.long),
        )
        meta = ShardMetadata(
            shard_id=shard_counter,
            file_name=f"{split_name}/{shard_filename}",
            num_samples=len(buffer_images),
            family_counts=dict(buffer_family_counts),
        )
        shards_metadata.append(meta)
        print(f"  [+] Persisted final {shard_filename} ({len(buffer_images)} samples).")

    return shards_metadata


async def run_dataset_generation(args: argparse.Namespace) -> None:
    """Execute complete Dataset V4 pipeline and persist catalog manifests."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    vocab_path = output_dir / "vocabulary_v4.json"
    print(f"[*] Building vocabulary at {vocab_path}...")
    vocabulary = _build_v4_vocabulary(vocab_path)
    print(f"[+] Vocabulary created with {len(vocabulary.token_to_index)} tokens.")

    splits_config: dict[str, int] = {
        "train": int(args.samples_per_family * 0.90),
        "val": int(args.samples_per_family * 0.05),
        "test": int(args.samples_per_family * 0.05),
    }

    all_shards: list[ShardMetadata] = []
    total_samples: int = 0

    for split_name, count_per_fam in splits_config.items():
        if count_per_fam > 0:
            split_shards = await generate_split_shards(
                split_name=split_name,
                output_dir=output_dir,
                samples_per_family=count_per_fam,
                shard_size=args.shard_size,
                vocabulary=vocabulary,
                concurrency=args.concurrency,
                seed=args.seed + (hash(split_name) % 1000),
                image_size=args.image_size,
                max_length=args.max_length,
            )
            all_shards.extend(split_shards)
            split_samples = sum(s.num_samples for s in split_shards)
            total_samples += split_samples

            # Write individual split manifest
            split_manifest = DatasetV4Manifest(
                version="v4",
                image_size=args.image_size,
                max_length=args.max_length,
                total_samples=split_samples,
                shards=split_shards,
            )
            with (output_dir / f"manifest_{split_name}.json").open("w", encoding="utf-8") as f:
                json.dump(split_manifest.to_dict(), f, indent=2)

    # Master manifest
    master_manifest = DatasetV4Manifest(
        version="v4",
        image_size=args.image_size,
        max_length=args.max_length,
        total_samples=total_samples,
        shards=all_shards,
    )
    master_manifest_path = output_dir / "manifest_v4.json"
    with master_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(master_manifest.to_dict(), f, indent=2)

    elapsed = time.time() - t_start
    print("\n[+] Dataset V4 Generation Complete!")
    print(f"[+] Total samples persisted: {total_samples}")
    print(f"[+] Shards count: {len(all_shards)}")
    throughput = total_samples / max(1.0, elapsed)
    print(f"[+] Total elapsed time: {elapsed:.2f}s ({throughput:.1f} samples/sec)")
    print(f"[+] Master manifest saved at: {master_manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sharded Dataset V4 at 256x256.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/v4_shards",
        help="Directory to persist shards and manifests.",
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=30000,
        help="Target samples per geometric family (default: 30000 -> 240,000 total).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of samples per shard file (default: 10000).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Spatial resolution (default: 256).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=max(2, (os.cpu_count() or 4)),
        help="Parallel rendering workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    asyncio.run(run_dataset_generation(args))


if __name__ == "__main__":
    main()
