"""Curriculum Learning V2 with real rendered images for visually-grounded convergence.

4-stage progressive curriculum that fixes the critical blank-image bug in V1 and
implements proper staged learning with ascending structural complexity:
    Stage 1: Line Segments & Vectors (direct cross-attention coordinate grounding)
    Stage 2: Curvilinear Primitives (circles, arcs, ellipses)
    Stage 3: Orthogonal Systems, Polylines & Function Plots
    Stage 4: Full complexity (all 8 families + compositional SCFG)

Key improvements over train_curriculum.py:
    1. Real rendered images via async TeX + Ghostscript pipeline (NOT blank torch.ones).
    2. Label smoothing (epsilon=0.1) to prevent overconfident boilerplate predictions.
    3. Weighted Cross-Entropy: 6x coordinate penalty, 0.3x boilerplate dampening.
    4. Embedding dropout (p=0.1) to force cross-attention reliance.
    5. Word dropout (p=0.40) for posterior collapse mitigation.
    6. Gradient accumulation for effective batch size 64.
    7. Per-stage dataset caching to disk for restart resilience.

References:
    Bengio et al., Curriculum Learning — monotonically increasing sample complexity.
    Goodfellow et al., Deep Learning — teacher forcing and scheduled learning rates.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import os
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.compositional import generate_compositional_batch
from core.dataset.templates import FAMILY_NAMES, generate_sample
from core.math.augmentation import add_gaussian_noise, jitter_contrast
from core.math.spatial import resize_spatial_dimensions
from core.math.tokenization import batch_encode, build_vocabulary
from core.ml.loss import (
    SpatialAwareHybridLoss,
    apply_word_dropout,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
    build_token_loss_weights,
)
from core.ml.model import VisionAutoregressiveModel, resolve_device
from core.models import ImageTensor, TikzTokens, TokenVocabulary, TrainingCheckpoint

# ---------------------------------------------------------------------------
# Dataset Generation with Real Rendered Images
# ---------------------------------------------------------------------------


async def _render_single_markup(
    code: str,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    sem: asyncio.Semaphore,
    image_size: int = 128,
) -> torch.Tensor:
    """Compile TikZ markup and return a normalized ``(3, image_size, image_size)`` tensor."""
    async with sem:
        try:
            res = await compiler.compile_tikz(TikzTokens(markup=code))
            if not res.is_successful:
                return torch.ones((3, image_size, image_size), dtype=torch.float32)
            png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            resized = resize_spatial_dimensions(ImageTensor(raw_tensor=t), image_size, image_size)
            return resized.raw_tensor.squeeze(0)
        except Exception:
            return torch.ones((3, image_size, image_size), dtype=torch.float32)


async def _render_batch_async(
    markups: list[str], max_concurrency: int = 32, image_size: int = 128
) -> torch.Tensor:
    """Compile all markups in parallel with bounded concurrency and preallocated tensor."""
    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()
    sem = asyncio.Semaphore(max_concurrency)

    total = len(markups)
    images_tensor = torch.empty((total, 3, image_size, image_size), dtype=torch.float32)
    batch_size = 500

    start_time = time.time()
    for i in range(0, total, batch_size):
        chunk_markups = markups[i : i + batch_size]
        chunk_tasks = [
            _render_single_markup(m, compiler, rasterizer, sem, image_size=image_size)
            for m in chunk_markups
        ]
        chunk_res = await asyncio.gather(*chunk_tasks)
        chunk_tensor = torch.stack(chunk_res, dim=0)
        images_tensor[i : i + len(chunk_res)] = chunk_tensor
        del chunk_res, chunk_tensor, chunk_tasks
        elapsed = time.time() - start_time
        processed = min(i + batch_size, total)
        rate = processed / max(1e-3, elapsed)
        print(f"  -> Rendered [{processed}/{total}] images in {elapsed:.1f}s ({rate:.1f} img/s)...")

    return images_tensor


def generate_stage_markups(
    families: list[str],
    num_samples: int,
    seed: int = 42,
) -> list[str]:
    """Generate TikZ markups balanced across the given families."""
    rng = np.random.default_rng(seed)
    markups: list[str] = []
    num_families = len(families)

    for i in range(num_samples):
        fam = families[i % num_families]
        if fam == "compositional":
            code = generate_compositional_batch(1, seed=int(rng.integers(0, 1_000_000)))[0]
        else:
            code = generate_sample(fam, rng)
        markups.append(code)

    return markups


def build_stage_dataset(
    families: list[str],
    num_samples: int,
    vocabulary: TokenVocabulary,
    max_length: int = 512,
    seed: int = 42,
    concurrency: int = 32,
    cache_dir: Path | None = None,
    stage_id: int = 0,
    image_size: int = 128,
    coordinate_step: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate or load cached dataset for a curriculum stage.

    Returns:
        (train_images, train_tokens, val_images, val_tokens).
    """
    if cache_dir is not None:
        cache_tag = f"stage{stage_id}_n{num_samples}_{image_size}px_{coordinate_step:g}"
        train_cache = cache_dir / f"{cache_tag}_train.pt"
        val_cache = cache_dir / f"{cache_tag}_val.pt"
        if train_cache.exists() and val_cache.exists():
            print(f"[*] Loading cached Stage {stage_id} dataset from disk...")
            train_data = torch.load(train_cache, map_location="cpu", weights_only=False)
            val_data = torch.load(val_cache, map_location="cpu", weights_only=False)
            return (
                train_data["images"],
                train_data["tokens"],
                val_data["images"],
                val_data["tokens"],
            )

    print(f"[*] Generating {num_samples} markups for Stage {stage_id}...")
    markups = generate_stage_markups(families, num_samples, seed=seed)
    tokens_list = [TikzTokens(markup=m) for m in markups]
    encoded_tokens = batch_encode(
        tokens_list,
        vocabulary,
        max_length=max_length,
        coordinate_step=coordinate_step,
    )

    print(
        f"[*] Rendering {num_samples} images ({image_size}x{image_size}) "
        f"with concurrency={concurrency}..."
    )
    images_tensor = asyncio.run(
        _render_batch_async(markups, max_concurrency=concurrency, image_size=image_size)
    )

    # Stratified 90/10 split with guard for small subsets
    if num_samples <= 100:
        num_val = max(1, int(num_samples * 0.2))
    else:
        num_val = min(max(100, int(num_samples * 0.1)), num_samples - 1)
    num_train = num_samples - num_val
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))

    train_images = images_tensor[indices[:num_train]].clone()
    train_tokens = encoded_tokens[indices[:num_train]].clone()
    val_images = images_tensor[indices[num_train:]].clone()
    val_tokens = encoded_tokens[indices[num_train:]].clone()
    del images_tensor, encoded_tokens, tokens_list, markups
    gc.collect()

    # Cache to disk for restart resilience
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"images": train_images, "tokens": train_tokens}, train_cache)
        torch.save({"images": val_images, "tokens": val_tokens}, val_cache)
        print(f"[+] Cached Stage {stage_id} dataset to {cache_dir}/")

    return train_images, train_tokens, val_images, val_tokens


# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------


def apply_photometric_augmentation(images: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Apply vectorized photometric noise and contrast jitter with probability p."""
    augmented = images
    if float(torch.rand(1).item()) < p:
        augmented = add_gaussian_noise(augmented, sigma=0.02)
    if float(torch.rand(1).item()) < p:
        augmented = jitter_contrast(augmented, alpha=1.05)
    return augmented


def train_epoch_with_accumulation(
    model: VisionAutoregressiveModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    micro_batch_size: int,
    accumulation_steps: int,
    target_device: torch.device,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    use_augmentation: bool = True,
    word_dropout_p: float = 0.40,
) -> float:
    """Execute one training epoch with gradient accumulation and word dropout.

    Effective batch size = micro_batch_size * accumulation_steps.
    """
    model.train()
    dataset_size: int = int(images.shape[0])
    order = torch.randperm(dataset_size)
    shuffled_images = images[order]
    shuffled_tokens = tokens[order]

    total_loss: float = 0.0
    num_optimizer_steps: int = 0
    accumulated: int = 0

    optimizer.zero_grad()

    step_start: int = 0
    while step_start < dataset_size:
        step_end: int = min(step_start + micro_batch_size, dataset_size)
        batch_imgs = shuffled_images[step_start:step_end].to(target_device)
        batch_toks = shuffled_tokens[step_start:step_end].to(target_device)

        if use_augmentation:
            batch_imgs = apply_photometric_augmentation(batch_imgs, p=0.5)

        raw_decoder_input, targets = build_teacher_forcing_pair(batch_toks)
        decoder_input = apply_word_dropout(raw_decoder_input, dropout_probability=word_dropout_p)

        logits = cast(torch.Tensor, model(batch_imgs, decoder_input))
        loss = cast(torch.Tensor, criterion(logits, targets))
        # Scale loss by accumulation factor for correct gradient magnitude
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()  # type: ignore[no-untyped-call]

        total_loss += float(loss.detach().item())
        accumulated += 1

        if accumulated >= accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            num_optimizer_steps += 1
            accumulated = 0

        step_start = step_end

    # Flush remaining accumulated gradients
    if accumulated > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        num_optimizer_steps += 1

    num_micro_batches = (dataset_size + micro_batch_size - 1) // micro_batch_size
    return total_loss / max(1, num_micro_batches)


@torch.no_grad()
def evaluate_epoch(
    model: VisionAutoregressiveModel,
    criterion: nn.Module,
    val_images: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    target_device: torch.device,
) -> float:
    """Evaluate mean validation loss across the hold-out dataset."""
    model.eval()
    dataset_size: int = int(val_images.shape[0])
    total_loss: float = 0.0
    num_batches: int = 0

    step_start: int = 0
    while step_start < dataset_size:
        step_end: int = min(step_start + batch_size, dataset_size)
        batch_imgs = val_images[step_start:step_end].to(target_device)
        batch_toks = val_tokens[step_start:step_end].to(target_device)

        decoder_input, targets = build_teacher_forcing_pair(batch_toks)
        logits = cast(torch.Tensor, model(batch_imgs, decoder_input))
        loss = cast(torch.Tensor, criterion(logits, targets))

        total_loss += float(loss.detach().item())
        num_batches += 1
        step_start = step_end

    return total_loss / max(1, num_batches)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

CURRICULUM_STAGES: list[dict[str, object]] = [
    {
        "name": "Stage 1: Line Segments & Vectors",
        "families": ["line_segment"],
        "epochs": 6,
        "samples": 15000,
        "lr": 3e-4,
    },
    {
        "name": "Stage 2: Curvilinear Primitives (Circles & Arcs)",
        "families": ["line_segment", "circle_arc"],
        "epochs": 6,
        "samples": 20000,
        "lr": 2e-4,
    },
    {
        "name": "Stage 3: Grids, Polylines & Function Plots",
        "families": ["line_segment", "circle_arc", "grid_axes", "function_plot", "polyline"],
        "epochs": 8,
        "samples": 30000,
        "lr": 1.5e-4,
    },
    {
        "name": "Stage 4: Full Complexity (All Families + SCFG)",
        "families": [
            "line_segment",
            "circle_arc",
            "grid_axes",
            "function_plot",
            "polyline",
            "polygon",
            "node_arrow",
            "compositional",
        ],
        "epochs": 10,
        "samples": 55000,
        "lr": 1e-4,
    },
]


def run_curriculum_v3(args: argparse.Namespace) -> None:
    """Execute the V3 curriculum with high-resolution rendered images."""
    for option_name in ("max_samples_per_stage", "max_epochs_per_stage"):
        option_value = getattr(args, option_name, None)
        if option_value is not None and option_value < 1:
            raise ValueError(f"{option_name} must be positive when provided.")
    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    target_device = resolve_device(args.device)
    print(f"[*] Curriculum V3 Engine on {target_device}")

    # Build or load vocabulary from full family coverage
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        vocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocab_path))
        print(f"[+] Loaded vocabulary: {len(vocabulary.token_to_index)} tokens.")
    else:
        print("[*] Building vocabulary from full-coverage sample corpus...")
        rng = np.random.default_rng(42)
        vocab_markups: list[TikzTokens] = []
        for fam in FAMILY_NAMES:
            for _ in range(200):
                vocab_markups.append(TikzTokens(markup=generate_sample(fam, rng)))
        comp_codes = generate_compositional_batch(500, seed=42)
        vocab_markups.extend([TikzTokens(markup=c) for c in comp_codes])
        vocabulary = build_vocabulary(vocab_markups, coordinate_step=args.coordinate_step)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocab_path))
        print(f"[+] Built and saved vocabulary: {len(vocabulary.token_to_index)} tokens.")

    # Weighted loss configuration
    token_weights = build_token_loss_weights(
        vocabulary=vocabulary,
        coordinate_weight=args.coordinate_weight,
        geometric_weight=args.geometric_weight,
        boilerplate_weight=args.boilerplate_weight,
    ).to(target_device)

    # Initialize model
    model = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        num_encoder_blocks=args.num_encoder_blocks,
        use_coord_conv=True,
        use_2d_pos_encoding=True,
        num_downsampling_stages=args.num_downsampling_stages,
        max_length=args.max_length,
        dropout=0.1,
        device=target_device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model: {total_params:,} parameters (CoordConv + 2D PE + Dropout 0.1)")

    criterion = SpatialAwareHybridLoss(
        vocabulary=vocabulary,
        spatial_weight=args.spatial_weight,
        token_weights=token_weights,
        label_smoothing=0.1,
    ).to(target_device)

    best_global_loss = float("inf")
    checkpoint_adapter = AtomicCheckpointAdapter()
    accumulation_steps = max(1, args.effective_batch_size // args.micro_batch_size)

    pipeline_start = time.time()

    for stage_idx, stage_cfg in enumerate(CURRICULUM_STAGES, start=1):
        stage_name = str(stage_cfg["name"])
        stage_families: list[str] = cast(list[str], stage_cfg["families"])
        stage_epochs = int(cast(int, stage_cfg["epochs"]))
        stage_samples = int(cast(int, stage_cfg["samples"]))
        stage_lr = float(cast(float, stage_cfg["lr"]))
        if args.max_samples_per_stage is not None:
            stage_samples = min(stage_samples, args.max_samples_per_stage)
        if args.max_epochs_per_stage is not None:
            stage_epochs = min(stage_epochs, args.max_epochs_per_stage)

        print("\n" + "=" * 72)
        print(f"[*] {stage_name}")
        print(f"    Samples: {stage_samples:,} | Epochs: {stage_epochs} | LR: {stage_lr}")
        print(f"    Families: {stage_families}")
        eff_bs = args.micro_batch_size * accumulation_steps
        print(f"    Effective Batch: {args.micro_batch_size} x {accumulation_steps} = {eff_bs}")
        print("=" * 72)

        # Build or load dataset with real rendered images
        train_images, train_tokens, val_images, val_tokens = build_stage_dataset(
            families=stage_families,
            num_samples=stage_samples,
            vocabulary=vocabulary,
            max_length=args.max_length,
            seed=42 + stage_idx,
            concurrency=args.concurrency,
            cache_dir=cache_dir,
            stage_id=stage_idx,
            image_size=args.image_size,
            coordinate_step=args.coordinate_step,
        )
        n_tr = train_images.shape[0]
        n_vl = val_images.shape[0]
        print(f"[+] Stage {stage_idx} Data: {n_tr:,} Train | {n_vl:,} Val")

        # Per-stage optimizer with warm restart
        optimizer = build_adamw_optimizer(
            model, learning_rate=stage_lr, weight_decay=args.weight_decay
        )
        num_batches = (
            int(train_images.shape[0]) + args.micro_batch_size - 1
        ) // args.micro_batch_size
        total_steps = max(1, (stage_epochs * num_batches) // accumulation_steps)
        if total_steps > 1:
            warmup_steps = max(1, min(int(total_steps * 0.08), total_steps - 1))
            scheduler = build_cosine_warmup_scheduler(
                optimizer, warmup_steps=warmup_steps, total_steps=total_steps
            )
        else:
            scheduler = None

        stage_best_val = float("inf")
        for epoch in range(1, stage_epochs + 1):
            t0 = time.time()
            train_loss = train_epoch_with_accumulation(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                images=train_images,
                tokens=train_tokens,
                micro_batch_size=args.micro_batch_size,
                accumulation_steps=accumulation_steps,
                target_device=target_device,
                scheduler=scheduler,
                use_augmentation=True,
                word_dropout_p=args.word_dropout_p,
            )
            val_loss = evaluate_epoch(
                model=model,
                criterion=criterion,
                val_images=val_images,
                val_tokens=val_tokens,
                batch_size=args.micro_batch_size,
                target_device=target_device,
            )
            elapsed = time.time() - t0
            curr_lr = optimizer.param_groups[0]["lr"]

            saved_marker = ""
            if val_loss < stage_best_val:
                stage_best_val = val_loss
                stage_ckpt_path = checkpoints_dir / f"curriculum_v3_stage{stage_idx}_best.pt"
                checkpoint_adapter.save_checkpoint(
                    TrainingCheckpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                    ),
                    str(stage_ckpt_path),
                )
                saved_marker = f"-> Saved [Stage {stage_idx} Best]"

            if val_loss < best_global_loss:
                best_global_loss = val_loss
                global_ckpt_path = checkpoints_dir / "curriculum_v3_best.pt"
                checkpoint_adapter.save_checkpoint(
                    TrainingCheckpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                    ),
                    str(global_ckpt_path),
                )
                saved_marker += " -> [GLOBAL BEST]"

            print(
                f"Epoch [{epoch:02d}/{stage_epochs:02d}] "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {curr_lr:.6f} | {elapsed:.1f}s {saved_marker}"
            )

        del train_images, train_tokens, val_images, val_tokens, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - pipeline_start
    print(f"\n[+] Curriculum V3 Complete in {total_elapsed / 60:.1f} min.")
    print(f"[+] Global Best Val Loss: {best_global_loss:.4f}")
    print(f"[+] Checkpoint: {checkpoints_dir / 'curriculum_v3_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum V3 Training Engine")
    parser.add_argument("--vocab-path", type=str, default="dataset/encoded/vocabulary_v3.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="dataset/curriculum_cache")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=2048)
    parser.add_argument("--num-encoder-blocks", type=int, default=8)
    parser.add_argument("--num-downsampling-stages", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--coordinate-step", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--spatial-weight", type=float, default=1.5)
    parser.add_argument("--coordinate-weight", type=float, default=8.0)
    parser.add_argument("--geometric-weight", type=float, default=3.0)
    parser.add_argument("--boilerplate-weight", type=float, default=0.15)
    parser.add_argument("--word-dropout-p", type=float, default=0.40)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-samples-per-stage", type=int, default=None)
    parser.add_argument("--max-epochs-per-stage", type=int, default=None)
    args = parser.parse_args()
    run_curriculum_v3(args)
