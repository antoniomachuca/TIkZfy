"""Procedural dataset generator and parallel tensor encoder for multi-tier training.

Generates balanced canonical and compositional SCFG samples, tokenizes them,
and rasterizes 64x64 normalized float tensors with high-throughput async batches.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset import FAMILY_NAMES, generate_batch, generate_compositional_batch
from core.math.spatial import resize_spatial_dimensions
from core.math.tokenization import batch_encode, build_vocabulary
from core.models import ImageTensor, TikzTokens


async def render_single_markup(
    code: str,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    sem: asyncio.Semaphore,
) -> torch.Tensor:
    """Compile TikZ markup and return a (3, 64, 64) normalized float tensor."""
    async with sem:
        try:
            res = await compiler.compile_tikz(TikzTokens(markup=code))
            if not res.is_successful:
                return torch.ones((3, 64, 64), dtype=torch.float32)
            png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            resized = resize_spatial_dimensions(ImageTensor(raw_tensor=t), 64, 64)
            return resized.raw_tensor.squeeze(0)
        except Exception:
            return torch.ones((3, 64, 64), dtype=torch.float32)


async def render_all_markups_async(
    markups: list[str], max_concurrency: int = 32
) -> list[torch.Tensor]:
    """Compile all markups in parallel with bounded concurrency."""
    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()
    sem = asyncio.Semaphore(max_concurrency)

    tasks = [render_single_markup(m, compiler, rasterizer, sem) for m in markups]

    total = len(tasks)
    batch_size = 500
    results: list[torch.Tensor] = []

    start_time = time.time()
    for i in range(0, total, batch_size):
        chunk = tasks[i : i + batch_size]
        chunk_res = await asyncio.gather(*chunk)
        results.extend(chunk_res)
        elapsed = time.time() - start_time
        processed = min(i + batch_size, total)
        rate = processed / max(1e-3, elapsed)
        print(f"  -> Rendered [{processed}/{total}] images in {elapsed:.1f}s ({rate:.1f} img/s)...")

    return results


def build_and_save_dataset(
    output_dir: str = "dataset/encoded",
    per_family: int = 1500,
    compositional_count: int = 3000,
    max_length: int = 512,
    seed: int = 42,
    concurrency: int = 32,
) -> None:
    """Generate multi-tier markups, tokenize, render tensors, and save to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[*] Generating procedural and compositional TikZ markups...")
    all_markups: list[str] = []

    # 8 Canonical Families
    for idx, fam in enumerate(FAMILY_NAMES):
        batch = generate_batch(fam, per_family, seed + idx)
        all_markups.extend(batch)
        print(f"  -> Generated {len(batch)} samples for {fam}")

    # Compositional SCFG
    comp_batch = generate_compositional_batch(compositional_count, seed=seed)
    all_markups.extend(comp_batch)
    print(f"  -> Generated {len(comp_batch)} compositional SCFG samples")

    total_samples = len(all_markups)
    print(f"[*] Total dataset size: {total_samples} samples.")

    # Build and save vocabulary
    tokens_list = [TikzTokens(markup=m) for m in all_markups]
    vocab = build_vocabulary(tokens_list)
    vocab_path = out_path / "vocabulary.json"
    JsonVocabularyAdapter().save_vocabulary(vocab, str(vocab_path))
    print(f"[*] Saved vocabulary with {len(vocab.token_to_index)} tokens to '{vocab_path}'.")

    # Encode token sequences
    print("[*] Encoding token sequences...")
    encoded_tokens = batch_encode(tokens_list, vocab, max_length=max_length)

    # Parallel Render images
    print(f"[*] Rendering {total_samples} image tensors (64x64) with concurrency={concurrency}...")
    rendered_images = asyncio.run(
        render_all_markups_async(all_markups, max_concurrency=concurrency)
    )
    images_tensor = torch.stack(rendered_images, dim=0)  # Shape: (N, 3, 64, 64)

    # Stratified split: 90% train, 10% val
    num_train = int(0.9 * total_samples)
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(seed))

    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_images = images_tensor[train_idx]
    train_tokens = encoded_tokens[train_idx]
    val_images = images_tensor[val_idx]
    val_tokens = encoded_tokens[val_idx]

    # Save to disk
    torch.save(train_images, out_path / "train_images.pt")
    torch.save(train_tokens, out_path / "train_tokens.pt")
    torch.save(val_images, out_path / "val_images.pt")
    torch.save(val_tokens, out_path / "val_tokens.pt")

    print(f"[+] Dataset successfully saved to '{output_dir}':")
    print(f"    Train: {train_images.shape[0]} samples | Val: {val_images.shape[0]} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="dataset/encoded")
    parser.add_argument("--per-family", type=int, default=1500)
    parser.add_argument("--compositional", type=int, default=3000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    build_and_save_dataset(
        output_dir=args.output_dir,
        per_family=args.per_family,
        compositional_count=args.compositional,
        max_length=args.max_length,
        concurrency=args.concurrency,
    )
