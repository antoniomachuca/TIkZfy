"""Curriculum Learning V2 with real rendered images for visually-grounded convergence.

4-stage progressive curriculum with ascending structural complexity:
    Stage 1: Line Segments & Vectors (15,000 samples)
    Stage 2: Curvilinear Primitives (20,000 samples)
    Stage 3: Orthogonal Systems, Polylines & Function Plots (30,000 samples)
    Stage 4: Full complexity (all 8 families + compositional SCFG) (55,000 samples)
    Total: 120,000 samples.

Phase 1 compliance features:
    1. Real rendered images via async TeX + Ghostscript pipeline (NOT blank torch.ones).
    2. Strict verification of tensor shapes (3, 128, 128) and non-blank images (std >= 1e-4).
    3. Per-sample JSONL manifest with exact seed, markup, family, and image SHA-256 hash.
    4. Deterministic SHA-256 hashing for vocabulary, dataset caches, and all checkpoints.
    5. Dual stream logging to stdout and train_v3_phase1_retrained.log.
    6. Isolated results and cache directory hierarchy to prevent artifact collisions.

References:
    Bengio et al., Curriculum Learning — monotonically increasing sample complexity.
    Goodfellow et al., Deep Learning — teacher forcing and scheduled learning rates.
    Golub & Van Loan, Matrix Computations — vectorized loss computation.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import io
import json
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
# Logging & Hashing Utilities
# ---------------------------------------------------------------------------


class DualLogger:
    """Synchronous tee logger writing formatted stream records to stdout and disk."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path: Path | None = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("a", encoding="utf-8")
        else:
            self._handle = None

    def log(self, message: str) -> None:
        """Write timestamped log message to stdout and optional log file."""
        timestamp: str = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted: str = f"[{timestamp}] {message}"
        print(formatted, flush=True)
        if self._handle is not None:
            self._handle.write(formatted + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Close log file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        chunk = handle.read(65536)
        while chunk:
            hasher.update(chunk)
            chunk = handle.read(65536)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Dataset Generation with Real Rendered Images & Manifest Tracking
# ---------------------------------------------------------------------------


async def _render_single_sample_robust(
    family: str,
    initial_seed: int,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    sem: asyncio.Semaphore,
    image_size: int = 128,
    max_retries: int = 10,
) -> tuple[torch.Tensor, str, float, str, int]:
    """Compile TikZ markup with deterministic fallback regeneration on edge-case degeneracies."""
    current_seed: int = initial_seed
    attempt: int = 0
    success: bool = False
    final_image: torch.Tensor = torch.empty(0)
    final_hash: str = ""
    final_std: float = 0.0
    final_markup: str = ""

    while attempt < max_retries and not success:
        sample_rng = np.random.default_rng(current_seed)
        if family == "composed":
            markup = generate_compositional_batch(1, seed=current_seed)[0]
        else:
            markup = generate_sample(family, sample_rng)

        async with sem:
            try:
                res = await compiler.compile_tikz(TikzTokens(markup=markup))
                if res.is_successful:
                    png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    arr = np.asarray(img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
                    resized = resize_spatial_dimensions(ImageTensor(raw_tensor=t), image_size, image_size)
                    image = resized.raw_tensor.squeeze(0)  # Shape: (3, H, W)
                    if image.shape == (3, image_size, image_size):
                        tensor_std = float(image.std().item())
                        if tensor_std >= 1e-4:
                            final_image = image
                            final_hash = hashlib.sha256(image.numpy().tobytes()).hexdigest()
                            final_std = tensor_std
                            final_markup = markup
                            success = True
            except Exception:
                pass

        if not success:
            current_seed = current_seed + 1000003
            attempt += 1

    if not success:
        raise RuntimeError(
            f"Failed to generate a valid non-degenerate sample for family '{family}' after {max_retries} attempts."
        )

    return final_image, final_hash, final_std, final_markup, current_seed


async def _render_stage_batch_async(
    families_list: list[str],
    seeds_list: list[int],
    logger: DualLogger,
    max_concurrency: int = 32,
    image_size: int = 128,
) -> tuple[torch.Tensor, list[str], list[float], list[str], list[int]]:
    """Compile markups in parallel with bounded concurrency, fallback retry, and manifest metrics."""
    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()
    sem = asyncio.Semaphore(max_concurrency)

    total: int = len(families_list)
    images_tensor: torch.Tensor = torch.empty((total, 3, image_size, image_size), dtype=torch.float32)
    image_hashes: list[str] = [""] * total
    tensor_stds: list[float] = [0.0] * total
    final_markups: list[str] = [""] * total
    final_seeds: list[int] = [0] * total
    batch_size: int = 500

    start_time: float = time.time()
    i: int = 0
    while i < total:
        chunk_end: int = min(i + batch_size, total)
        chunk_fams: list[str] = families_list[i:chunk_end]
        chunk_seeds: list[int] = seeds_list[i:chunk_end]
        chunk_tasks = [
            _render_single_sample_robust(
                fam, seed, compiler, rasterizer, sem, image_size=image_size
            )
            for fam, seed in zip(chunk_fams, chunk_seeds, strict=True)
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)
        for offset, (img_t, img_hash, img_std, f_markup, f_seed) in enumerate(chunk_results):
            images_tensor[i + offset] = img_t
            image_hashes[i + offset] = img_hash
            tensor_stds[i + offset] = img_std
            final_markups[i + offset] = f_markup
            final_seeds[i + offset] = f_seed
        del chunk_results, chunk_tasks
        elapsed: float = time.time() - start_time
        rate: float = chunk_end / max(1e-3, elapsed)
        logger.log(f"  -> Rendered [{chunk_end}/{total}] images in {elapsed:.1f}s ({rate:.1f} img/s)...")
        i = chunk_end

    return images_tensor, image_hashes, tensor_stds, final_markups, final_seeds


def generate_stage_seeds(
    families: list[str],
    num_samples: int,
    base_seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Generate family tags and deterministic base seeds balanced across families."""
    rng = np.random.default_rng(base_seed)
    sample_families: list[str] = []
    sample_seeds: list[int] = []
    num_families: int = len(families)

    for i in range(num_samples):
        fam: str = families[i % num_families]
        sample_seed: int = int(rng.integers(0, 2_000_000_000))
        sample_families.append(fam)
        sample_seeds.append(sample_seed)

    return sample_families, sample_seeds


def build_stage_dataset(
    families: list[str],
    num_samples: int,
    vocabulary: TokenVocabulary,
    logger: DualLogger,
    max_length: int = 512,
    seed: int = 42,
    concurrency: int = 32,
    cache_dir: Path | None = None,
    manifest_dir: Path | None = None,
    stage_id: int = 0,
    image_size: int = 128,
    coordinate_step: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate or load cached dataset for a curriculum stage with reproducible manifest."""
    cache_tag: str = f"stage{stage_id}_n{num_samples}_{image_size}px_{coordinate_step:g}"
    train_cache: Path | None = cache_dir / f"{cache_tag}_train.pt" if cache_dir is not None else None
    val_cache: Path | None = cache_dir / f"{cache_tag}_val.pt" if cache_dir is not None else None

    if train_cache is not None and val_cache is not None and train_cache.exists() and val_cache.exists():
        logger.log(f"[*] Loading cached Stage {stage_id} dataset from disk...")
        train_data = torch.load(train_cache, map_location="cpu", weights_only=False)
        val_data = torch.load(val_cache, map_location="cpu", weights_only=False)
        train_hash: str = compute_file_sha256(train_cache)
        val_hash: str = compute_file_sha256(val_cache)
        logger.log(f"    Train Cache SHA-256: {train_hash}")
        logger.log(f"    Val Cache SHA-256:   {val_hash}")
        return (
            train_data["images"],
            train_data["tokens"],
            val_data["images"],
            val_data["tokens"],
        )

    logger.log(f"[*] Generating and rendering {num_samples} samples for Stage {stage_id}...")
    sample_fams, sample_seeds = generate_stage_seeds(
        families, num_samples, base_seed=seed
    )

    images_tensor, image_hashes, tensor_stds, final_markups, final_seeds = asyncio.run(
        _render_stage_batch_async(
            families_list=sample_fams,
            seeds_list=sample_seeds,
            logger=logger,
            max_concurrency=concurrency,
            image_size=image_size,
        )
    )

    tokens_list: list[TikzTokens] = [TikzTokens(markup=m) for m in final_markups]
    encoded_tokens: torch.Tensor = batch_encode(
        tokens_list,
        vocabulary,
        max_length=max_length,
        coordinate_step=coordinate_step,
    )

    # Save reproducible manifest
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path: Path = manifest_dir / f"stage{stage_id}_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for idx in range(num_samples):
                record = {
                    "sample_id": f"stage{stage_id}_sample_{idx:06d}",
                    "stage": stage_id,
                    "family": sample_fams[idx],
                    "seed": final_seeds[idx],
                    "markup": final_markups[idx],
                    "tensor_shape": list(images_tensor[idx].shape),
                    "tensor_std": tensor_stds[idx],
                    "image_sha256": image_hashes[idx],
                    "compilation_status": "SUCCESS",
                }
                manifest_file.write(json.dumps(record) + "\n")
        manifest_hash: str = compute_file_sha256(manifest_path)
        logger.log(f"[+] Manifest persisted to {manifest_path} (SHA-256: {manifest_hash})")

    # Stratified 90/10 split
    if num_samples <= 100:
        num_val: int = max(1, int(num_samples * 0.2))
    else:
        num_val = min(max(100, int(num_samples * 0.1)), num_samples - 1)
    num_train: int = num_samples - num_val
    indices: torch.Tensor = torch.randperm(
        num_samples, generator=torch.Generator().manual_seed(seed)
    )

    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    if cache_dir is not None and train_cache is not None and val_cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"images": images_tensor[train_idx], "tokens": encoded_tokens[train_idx]}, train_cache)
        torch.save({"images": images_tensor[val_idx], "tokens": encoded_tokens[val_idx]}, val_cache)
        train_hash = compute_file_sha256(train_cache)
        val_hash = compute_file_sha256(val_cache)
        logger.log(f"[+] Cached Stage {stage_id} dataset to {cache_dir}/")
        logger.log(f"    Train Cache SHA-256: {train_hash}")
        logger.log(f"    Val Cache SHA-256:   {val_hash}")

    train_images: torch.Tensor = images_tensor[train_idx]
    train_tokens: torch.Tensor = encoded_tokens[train_idx]
    val_images: torch.Tensor = images_tensor[val_idx]
    val_tokens: torch.Tensor = encoded_tokens[val_idx]
    del images_tensor, encoded_tokens, tokens_list, final_markups
    gc.collect()

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
            "composed",
        ],
        "epochs": 10,
        "samples": 55000,
        "lr": 1e-4,
    },
]


def validate_curriculum_configuration() -> None:
    """Validate stage families and the planned 120,000-sample curriculum."""
    expected_families = set(FAMILY_NAMES)
    total_samples = 0
    for stage in CURRICULUM_STAGES:
        families = cast(list[str], stage["families"])
        unknown = set(families) - expected_families
        if unknown:
            raise ValueError(f"Curriculum contains unknown families: {sorted(unknown)}.")
        total_samples += int(cast(int, stage["samples"]))
    if total_samples != 120_000:
        raise ValueError(f"Curriculum must configure 120000 samples, got {total_samples}.")


def run_curriculum_v3(args: argparse.Namespace) -> None:
    """Execute the V3 curriculum with high-resolution rendered images and logging."""
    validate_curriculum_configuration()
    for option_name in ("max_samples_per_stage", "max_epochs_per_stage"):
        option_value = getattr(args, option_name, None)
        if option_value is not None and option_value < 1:
            raise ValueError(f"{option_name} must be positive when provided.")

    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    manifest_dir = results_dir / "manifests"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_path: Path = Path(args.log_file) if args.log_file is not None else results_dir / "train_v3_phase1_retrained.log"
    logger = DualLogger(log_path)

    target_device = resolve_device(args.device)
    logger.log(f"[*] Curriculum V3 Engine starting on {target_device}")
    logger.log(f"[*] Random Seed: {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build or load vocabulary from full family coverage
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        vocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocab_path))
        vocab_hash: str = compute_file_sha256(vocab_path)
        logger.log(f"[+] Loaded vocabulary: {len(vocabulary.token_to_index)} tokens (SHA-256: {vocab_hash})")
    else:
        logger.log("[*] Building vocabulary from full-coverage sample corpus...")
        rng = np.random.default_rng(args.seed)
        vocab_markups: list[TikzTokens] = []
        for fam in FAMILY_NAMES:
            for _ in range(200):
                vocab_markups.append(TikzTokens(markup=generate_sample(fam, rng)))
        comp_codes = generate_compositional_batch(500, seed=args.seed)
        vocab_markups.extend([TikzTokens(markup=c) for c in comp_codes])
        vocabulary = build_vocabulary(vocab_markups, coordinate_step=args.coordinate_step)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocab_path))
        vocab_hash = compute_file_sha256(vocab_path)
        logger.log(f"[+] Built and saved vocabulary: {len(vocabulary.token_to_index)} tokens (SHA-256: {vocab_hash})")

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
    total_params: int = sum(p.numel() for p in model.parameters())
    logger.log(f"[+] Model Architecture: {total_params:,} parameters (CoordConv + 2D PE + Dropout 0.1)")

    config_record = {
        "seed": args.seed,
        "device": str(target_device),
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_ff": args.dim_ff,
        "num_encoder_blocks": args.num_encoder_blocks,
        "num_downsampling_stages": args.num_downsampling_stages,
        "image_size": args.image_size,
        "coordinate_step": args.coordinate_step,
        "max_length": args.max_length,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "weight_decay": args.weight_decay,
        "spatial_weight": args.spatial_weight,
        "coordinate_weight": args.coordinate_weight,
        "geometric_weight": args.geometric_weight,
        "boilerplate_weight": args.boilerplate_weight,
        "word_dropout_p": args.word_dropout_p,
        "vocabulary_path": str(vocab_path),
        "vocabulary_sha256": vocab_hash,
        "vocabulary_size": len(vocabulary.token_to_index),
        "stages": CURRICULUM_STAGES,
    }
    with (results_dir / "experiment_config.json").open("w", encoding="utf-8") as f_cfg:
        json.dump(config_record, f_cfg, indent=2)
    logger.log(f"[*] Complete experiment configuration dumped to {results_dir / 'experiment_config.json'}")

    criterion = SpatialAwareHybridLoss(
        vocabulary=vocabulary,
        spatial_weight=args.spatial_weight,
        token_weights=token_weights,
        label_smoothing=0.1,
    ).to(target_device)

    best_global_loss: float = float("inf")
    checkpoint_adapter = AtomicCheckpointAdapter()
    accumulation_steps: int = max(1, args.effective_batch_size // args.micro_batch_size)

    pipeline_start: float = time.time()
    stage_summaries: list[dict[str, object]] = []

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

        logger.log("\n" + "=" * 72)
        logger.log(f"[*] {stage_name}")
        logger.log(f"    Samples: {stage_samples:,} | Epochs: {stage_epochs} | LR: {stage_lr}")
        logger.log(f"    Families: {stage_families}")
        eff_bs = args.micro_batch_size * accumulation_steps
        logger.log(f"    Effective Batch: {args.micro_batch_size} x {accumulation_steps} = {eff_bs}")
        logger.log("=" * 72)

        # Build or load dataset with real rendered images
        train_images, train_tokens, val_images, val_tokens = build_stage_dataset(
            families=stage_families,
            num_samples=stage_samples,
            vocabulary=vocabulary,
            logger=logger,
            max_length=args.max_length,
            seed=args.seed + stage_idx,
            concurrency=args.concurrency,
            cache_dir=cache_dir,
            manifest_dir=manifest_dir,
            stage_id=stage_idx,
            image_size=args.image_size,
            coordinate_step=args.coordinate_step,
        )
        n_tr: int = train_images.shape[0]
        n_vl: int = val_images.shape[0]
        logger.log(f"[+] Stage {stage_idx} Data Shape: Train {train_images.shape} | Val {val_images.shape}")

        # Per-stage optimizer with warm restart
        optimizer = build_adamw_optimizer(
            model, learning_rate=stage_lr, weight_decay=args.weight_decay
        )
        num_batches: int = (
            int(train_images.shape[0]) + args.micro_batch_size - 1
        ) // args.micro_batch_size
        total_steps: int = max(1, (stage_epochs * num_batches) // accumulation_steps)
        if total_steps > 1:
            warmup_steps = max(1, min(int(total_steps * 0.08), total_steps - 1))
            scheduler = build_cosine_warmup_scheduler(
                optimizer, warmup_steps=warmup_steps, total_steps=total_steps
            )
        else:
            scheduler = None

        stage_best_val: float = float("inf")
        stage_best_epoch: int = 0
        stage_t0: float = time.time()

        for epoch in range(1, stage_epochs + 1):
            t0: float = time.time()
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
            elapsed: float = time.time() - t0
            curr_lr: float = optimizer.param_groups[0]["lr"]

            saved_marker: str = ""
            if val_loss < stage_best_val:
                stage_best_val = val_loss
                stage_best_epoch = epoch
                stage_ckpt_path = checkpoints_dir / f"curriculum_v3_stage{stage_idx}_best.pt"
                checkpoint_adapter.save_checkpoint(
                    TrainingCheckpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                    ),
                    str(stage_ckpt_path),
                )
                ckpt_hash = compute_file_sha256(stage_ckpt_path)
                saved_marker = f"-> Saved [Stage {stage_idx} Best: {stage_ckpt_path.name} | SHA-256: {ckpt_hash}]"

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
                global_hash = compute_file_sha256(global_ckpt_path)
                saved_marker += f" -> [GLOBAL BEST: {global_ckpt_path.name} | SHA-256: {global_hash}]"

            logger.log(
                f"Epoch [{epoch:02d}/{stage_epochs:02d}] "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {curr_lr:.6f} | {elapsed:.1f}s {saved_marker}"
            )

        stage_elapsed: float = time.time() - stage_t0
        stage_summaries.append({
            "stage": stage_idx,
            "name": stage_name,
            "samples": stage_samples,
            "epochs": stage_epochs,
            "best_val_loss": stage_best_val,
            "best_epoch": stage_best_epoch,
            "elapsed_seconds": stage_elapsed,
        })

        del train_images, train_tokens, val_images, val_tokens, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed: float = time.time() - pipeline_start
    logger.log("\n" + "=" * 72)
    logger.log(f"[+] Curriculum V3 Training Complete in {total_elapsed / 60:.1f} min.")
    logger.log(f"[+] Global Best Val Loss: {best_global_loss:.4f}")
    logger.log(f"[+] Final Checkpoint: {checkpoints_dir / 'curriculum_v3_best.pt'}")
    logger.log(f"[+] Final Checkpoint SHA-256: {compute_file_sha256(checkpoints_dir / 'curriculum_v3_best.pt')}")
    logger.log("=" * 72)

    summary_path: Path = results_dir / "curriculum_v3_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump({
            "total_elapsed_seconds": total_elapsed,
            "global_best_val_loss": best_global_loss,
            "stages": stage_summaries,
        }, f_sum, indent=2)
    logger.log(f"[*] Training summary persisted to {summary_path}")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum V3 Training Engine")
    parser.add_argument("--vocab-path", type=str, default="dataset/encoded/vocabulary_v3.json")
    parser.add_argument("--results-dir", type=str, default="results/phase1_v3_retrained")
    parser.add_argument("--cache-dir", type=str, default="dataset/curriculum_cache_v3_retrained")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
