"""Master Phase 5 multi-task curriculum training orchestrator for V4 Image-to-TikZ engine.

Executes a 3-stage stratified additive curriculum on NVIDIA L4 GPU (24GB VRAM):
    Stage 1: 50% Simple, 30% Orthogonal, 20% Complex (10 epochs, lr=3.0e-4)
    Stage 2: 30% Simple, 30% Orthogonal/Curvilinear, 40% Complex (15 epochs, lr=2.0e-4)
    Stage 3: 12.5% Uniform (8 families) with photometric augmentation (15 epochs, lr=1.0e-4)

Key Features:
    - High-throughput streaming via ShardedTikzDataset over pre-encoded .pt shards.
    - Native bfloat16 mixed precision on 4th-generation Tensor Cores (NVIDIA L4).
    - Joint optimization with AdamW and CompositeMultiTaskLossV4.
    - Dual-stream logging to stdout and train_v4.log.
    - Atomic checkpoint persistence (global best, stage best, latest resume).
    - Real-time telemetry JSON export for remote monitoring.
    - Zero-cost sentinel auto-shutdown (sudo poweroff).

References:
    Bengio et al., Curriculum Learning — additive distribution shifts.
    Goodfellow et al., Deep Learning — multi-task cross-entropy and teacher forcing (§10.2.1).
    Golub & Van Loan, Matrix Computations — vectorized loss matrices and coordinate lattices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.utils.data import DataLoader, Sampler

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.sharded import ShardedTikzDataset
from core.math.augmentation import add_gaussian_noise, jitter_contrast
from core.ml.loss import (
    CompositeMultiTaskLossV4,
    LossComponents,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
)
from core.ml.model import VisionAutoregressiveModelV4, resolve_device
from core.models import (
    FAMILY_NAMES,
    TrainingCheckpoint,
)

# ---------------------------------------------------------------------------
# Curriculum Stage Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumStage:
    """Hyperparameter specification for a single curriculum stage."""

    stage_idx: int
    name: str
    num_epochs: int
    learning_rate: float
    # Relative sampling weights for the 8 canonical families:
    # 0: line_segment, 1: polyline, 2: polygon, 3: circle_arc,
    # 4: grid_axes, 5: function_plot, 6: node_arrow, 7: composed
    family_weights: tuple[float, ...]
    enable_augmentation: bool
    description: str


CANONICAL_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        stage_idx=1,
        name="Stage 1: Coordinate Anchoring & Simple Geometries",
        num_epochs=10,
        learning_rate=3.0e-4,
        # 50% Simple (line 0.25, circle 0.25)
        # 30% Orthogonal (poly 0.075, polygon 0.075, grid 0.075, plot 0.075)
        # 20% Complex (node 0.10, composed 0.10)
        family_weights=(0.25, 0.075, 0.075, 0.25, 0.075, 0.075, 0.10, 0.10),
        enable_augmentation=False,
        description="CoordConv stem initialization, coordinate anchoring, and basic syntax.",
    ),
    CurriculumStage(
        stage_idx=2,
        name="Stage 2: Topology Decoupling & Complex Systems",
        num_epochs=15,
        learning_rate=2.0e-4,
        # 30% Simple (0.15, 0.15)
        # 30% Orthogonal/Curvilinear (0.075, 0.075, 0.075, 0.075)
        # 40% Complex (0.20, 0.20)
        family_weights=(0.15, 0.075, 0.075, 0.15, 0.075, 0.075, 0.20, 0.20),
        enable_augmentation=False,
        description="Multi-task auxiliary family head specialization and complex node/SCFG graphs.",
    ),
    CurriculumStage(
        stage_idx=3,
        name="Stage 3: Global Uniform Consolidation & Robustness",
        num_epochs=15,
        learning_rate=1.0e-4,
        # 12.5% Uniform across all 8 families
        family_weights=(0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125),
        enable_augmentation=True,
        description="Fine-grained metric convergence with photometric noise and contrast jitter.",
    ),
)


# ---------------------------------------------------------------------------
# Logging & Telemetry Infrastructure
# ---------------------------------------------------------------------------


class DualLogger:
    """Synchronous logger writing formatted records to stdout and disk."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path: Path | None = log_path
        self._handle: TextIO | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        """Write timestamped log record."""
        timestamp: str = time.strftime("%Y-%m-%d %H:%M:%S")
        record: str = f"[{timestamp}] {message}"
        print(record, flush=True)
        if self._handle is not None:
            self._handle.write(record + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Flush and close underlying file descriptor."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass
class EpochTelemetry:
    """Per-epoch telemetry statistics for monitoring and visualization."""

    stage_idx: int
    epoch: int
    global_epoch: int
    train_loss: float
    train_syntax_loss: float
    train_gaussian_ord_loss: float
    train_huber_loss: float
    train_family_loss: float
    val_loss: float
    val_syntax_loss: float
    val_gaussian_ord_loss: float
    val_huber_loss: float
    val_family_loss: float
    val_family_acc: float
    learning_rate: float
    samples_per_sec: float
    gpu_memory_used_mb: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Stratified Curriculum Sampler
# ---------------------------------------------------------------------------


class StratifiedCurriculumSampler(Sampler[int]):
    """Draws sample indices obeying designated family probability distributions.

    Partitions dataset sample indices into 8 family pools, drawing proportionally
    to target curriculum weights without replacement within each epoch iteration.
    """

    def __init__(
        self,
        dataset: ShardedTikzDataset,
        family_weights: tuple[float, ...],
        total_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._num_samples: int = total_samples if total_samples is not None else len(dataset)
        self._family_weights: torch.Tensor = torch.tensor(family_weights, dtype=torch.float64)
        self._family_weights = self._family_weights / self._family_weights.sum()
        self._rng: torch.Generator = torch.Generator().manual_seed(seed)

        self._family_indices: list[list[int]] = [[] for _ in range(len(FAMILY_NAMES))]
        curr_idx = 0
        for shard in dataset.manifest.shards:
            for fam_name, count in shard.family_counts.items():
                if fam_name in FAMILY_NAMES:
                    fam_id = FAMILY_NAMES.index(fam_name)
                    self._family_indices[fam_id].extend(range(curr_idx, curr_idx + count))
                    curr_idx += count

        all_collected = sum(len(indices) for indices in self._family_indices)
        if all_collected != len(dataset):
            self._family_indices = [[] for _ in range(len(FAMILY_NAMES))]
            samples_per_fam = len(dataset) // len(FAMILY_NAMES)
            for f_id in range(len(FAMILY_NAMES)):
                start = f_id * samples_per_fam
                end = (f_id + 1) * samples_per_fam if f_id < 7 else len(dataset)
                self._family_indices[f_id] = list(range(start, end))

    def __iter__(self) -> Any:
        family_draws = torch.multinomial(
            self._family_weights,
            num_samples=self._num_samples,
            replacement=True,
            generator=self._rng,
        )

        permuted_pools: list[list[int]] = []
        for pool in self._family_indices:
            perm = torch.randperm(len(pool), generator=self._rng).tolist()
            permuted_pools.append([pool[i] for i in perm])

        pool_pointers: list[int] = [0] * len(FAMILY_NAMES)
        yielded_indices: list[int] = []

        for fam_tensor in family_draws:
            fam = int(fam_tensor.item())
            pool = permuted_pools[fam]
            ptr = pool_pointers[fam]
            if ptr >= len(pool):
                perm = torch.randperm(len(self._family_indices[fam]), generator=self._rng).tolist()
                permuted_pools[fam] = [self._family_indices[fam][i] for i in perm]
                ptr = 0
            yielded_indices.append(permuted_pools[fam][ptr])
            pool_pointers[fam] = ptr + 1

        return iter(yielded_indices)

    def __len__(self) -> int:
        return self._num_samples


# ---------------------------------------------------------------------------
# Training & Evaluation Epoch Functions
# ---------------------------------------------------------------------------


def run_training_epoch(
    model: VisionAutoregressiveModelV4,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]],
    criterion: CompositeMultiTaskLossV4,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    grad_accum_steps: int = 2,
    enable_augmentation: bool = False,
    logger: DualLogger | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Execute single training epoch with bfloat16 autocast and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_syntax = 0.0
    total_gaussian = 0.0
    total_huber = 0.0
    total_family = 0.0
    num_batches = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (raw_images, target_tokens, family_labels) in enumerate(dataloader):
        images: torch.Tensor = raw_images.to(device, non_blocking=True)
        tokens: torch.Tensor = target_tokens.to(device, non_blocking=True)
        families: torch.Tensor = family_labels.to(device, non_blocking=True)

        if enable_augmentation:
            images = jitter_contrast(images, alpha=1.1)
            images = add_gaussian_noise(images, sigma=0.015)

        decoder_input, targets = build_teacher_forcing_pair(tokens)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            token_logits, family_logits = model(images, decoder_input, return_family_logits=True)
            components: LossComponents = criterion(
                token_logits=token_logits,
                target_tokens=targets,
                family_logits=family_logits,
                family_targets=families,
                return_components=True,
            )
            scaled_loss = components.total_loss / grad_accum_steps

        scaled_loss.backward()  # type: ignore[no-untyped-call]

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += components.total_loss.detach().item()
        total_syntax += components.syntax_loss.detach().item()
        total_gaussian += components.gaussian_ord_loss.detach().item()
        total_huber += components.huber_loss.detach().item()
        total_family += components.family_loss.detach().item()
        num_batches += 1

        if (step + 1) % 250 == 0 and logger is not None:
            elapsed = time.time() - t0
            batch_sz = dataloader.batch_size if dataloader.batch_size is not None else 1
            rate = (step + 1) * batch_sz / max(1.0, elapsed)
            lr_val = optimizer.param_groups[0]["lr"]
            logger.log(
                f"  Step [{step + 1:>5}/{len(dataloader)}] "
                f"Loss: {total_loss / num_batches:.4f} "
                f"(Syn: {total_syntax / num_batches:.3f}, "
                f"Ord: {total_gaussian / num_batches:.3f}, "
                f"Hub: {total_huber / num_batches:.3f}, "
                f"Fam: {total_family / num_batches:.3f}) | "
                f"LR: {lr_val:.2e} | {rate:.1f} samples/s"
            )

    denom = max(1, num_batches)
    elapsed_total = time.time() - t0
    batch_sz_total = dataloader.batch_size if dataloader.batch_size is not None else 1
    samples_sec = (num_batches * batch_sz_total) / max(1.0, elapsed_total)
    return (
        total_loss / denom,
        total_syntax / denom,
        total_gaussian / denom,
        total_huber / denom,
        total_family / denom,
        samples_sec,
    )


def run_validation_epoch(
    model: VisionAutoregressiveModelV4,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]],
    criterion: CompositeMultiTaskLossV4,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate validation split and compute classification accuracy."""
    model.eval()
    total_loss = 0.0
    total_syntax = 0.0
    total_gaussian = 0.0
    total_huber = 0.0
    total_family = 0.0
    correct_families = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for raw_images, target_tokens, family_labels in dataloader:
            images = raw_images.to(device, non_blocking=True)
            tokens = target_tokens.to(device, non_blocking=True)
            families = family_labels.to(device, non_blocking=True)

            decoder_input, targets = build_teacher_forcing_pair(tokens)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                token_logits, family_logits = model(
                    images, decoder_input, return_family_logits=True
                )
                components: LossComponents = criterion(
                    token_logits=token_logits,
                    target_tokens=targets,
                    family_logits=family_logits,
                    family_targets=families,
                    return_components=True,
                )

            total_loss += components.total_loss.item()
            total_syntax += components.syntax_loss.item()
            total_gaussian += components.gaussian_ord_loss.item()
            total_huber += components.huber_loss.item()
            total_family += components.family_loss.item()
            num_batches += 1

            preds = family_logits.argmax(dim=-1)
            correct_families += int((preds == families).sum().item())
            total_samples += images.size(0)

    denom = max(1, num_batches)
    family_acc = (correct_families / max(1, total_samples)) * 100.0
    return (
        total_loss / denom,
        total_syntax / denom,
        total_gaussian / denom,
        total_huber / denom,
        total_family / denom,
        family_acc,
    )


# ---------------------------------------------------------------------------
# Master Orchestration Loop
# ---------------------------------------------------------------------------


def execute_v4_curriculum_training(args: argparse.Namespace) -> None:
    """Main entrypoint orchestrating data loading, 3-stage curriculum, and checkpoints."""
    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger = DualLogger(results_dir / "train_v4.log")
    logger.log("================================================================================")
    logger.log("    TIKZFY V4 MULTIMODAL CURRICULUM TRAINING (NVIDIA L4 24GB)")
    logger.log("================================================================================")

    device = resolve_device(args.device)
    logger.log(f"[*] Target Compute Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.log(f"[*] GPU Model: {props.name} | Total VRAM: {props.total_memory / 1e9:.2f} GB")

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        fallback = Path(args.data_dir) / "vocabulary_v4.json"
        if fallback.exists():
            vocab_path = fallback
        else:
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path} or {fallback}.")
    logger.log(f"[*] Loading vocabulary from {vocab_path}...")
    vocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocab_path))
    logger.log(f"[+] Vocabulary loaded: {len(vocabulary.token_to_index)} tokens.")

    data_dir = Path(args.data_dir)
    train_manifest = data_dir / "manifest_train.json"
    val_manifest = data_dir / "manifest_val.json"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    logger.log(f"[*] Initializing ShardedTikzDataset (cache_size={args.cache_size})...")
    train_dataset = ShardedTikzDataset(train_manifest, cache_size=args.cache_size)
    val_dataset = ShardedTikzDataset(val_manifest, cache_size=max(2, args.cache_size // 2))
    logger.log(
        f"[+] Datasets ready: {len(train_dataset):,} train samples, "
        f"{len(val_dataset):,} val samples."
    )

    logger.log("[*] Instantiating VisionAutoregressiveModelV4 (256x256, 1024 tokens)...")
    model = VisionAutoregressiveModelV4(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=args.model_dim,
        max_length=args.max_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        num_encoder_blocks=args.num_encoder_blocks,
        num_downsampling_stages=3,
        num_families=len(FAMILY_NAMES),
        dropout=args.dropout,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"[+] Model V4 Parameter Count: {total_params:,} trainable weights.")

    criterion = CompositeMultiTaskLossV4(
        vocabulary=vocabulary,
        lambda_coord=args.lambda_coord,
        lambda_family=args.lambda_family,
        lambda_spatial=args.lambda_spatial,
        label_smoothing=args.label_smoothing,
        sigma=args.sigma,
        huber_beta=args.huber_beta,
    ).to(device)

    checkpoint_adapter = AtomicCheckpointAdapter()
    best_global_val_loss = float("inf")
    start_global_epoch = 1
    telemetry_history: list[dict[str, Any]] = []

    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.log(f"[*] Resuming from checkpoint: {resume_path}...")
            cp = checkpoint_adapter.load_checkpoint(str(resume_path))
            model.load_state_dict(cp.model_state)
            start_global_epoch = cp.epoch + 1
            logger.log(f"[+] Resumed state: epoch={cp.epoch}.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    global_epoch_counter = start_global_epoch

    for stage in CANONICAL_STAGES:
        if stage.stage_idx < args.start_stage:
            logger.log(f"[-] Skipping {stage.name} (per --start-stage {args.start_stage}).")
            continue

        logger.log("\n" + "=" * 70)
        logger.log(f"    STARTING {stage.name.upper()}")
        logger.log(f"    {stage.description}")
        logger.log(f"    Epochs: {stage.num_epochs} | Base LR: {stage.learning_rate:.2e}")
        logger.log("=" * 70)

        sampler = StratifiedCurriculumSampler(
            dataset=train_dataset,
            family_weights=stage.family_weights,
            total_samples=args.max_samples_per_epoch,
            seed=args.seed + stage.stage_idx * 100,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        optimizer = build_adamw_optimizer(
            model,
            learning_rate=stage.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_training_steps = max(2, len(train_loader) * stage.num_epochs // args.grad_accum_steps)
        warmup_steps = max(1, min(50, total_training_steps // 10))
        if warmup_steps >= total_training_steps:
            warmup_steps = max(1, total_training_steps // 2)
        scheduler = build_cosine_warmup_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_training_steps,
            min_lr_ratio=0.05,
        )

        best_stage_val_loss = float("inf")

        for stage_epoch in range(1, stage.num_epochs + 1):
            t_epoch_start = time.time()
            logger.log(
                f"\n>>> Stage {stage.stage_idx} | Epoch [{stage_epoch:>2}/{stage.num_epochs}] "
                f"(Global Epoch {global_epoch_counter})"
            )

            (
                train_loss,
                train_syn,
                train_ord,
                train_hub,
                train_fam,
                samples_sec,
            ) = run_training_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                enable_augmentation=stage.enable_augmentation,
                logger=logger,
            )

            (
                val_loss,
                val_syn,
                val_ord,
                val_hub,
                val_fam,
                val_acc,
            ) = run_validation_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            elapsed = time.time() - t_epoch_start
            vram_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            )

            logger.log(
                f"[Summary] Train: {train_loss:.4f} | Val: {val_loss:.4f} "
                f"(Syn: {val_syn:.2f}, Ord: {val_ord:.2f}, Hub: {val_hub:.2f}, Fam: {val_fam:.2f}) "
                f"| Fam Acc: {val_acc:.1f}% | Rate: {samples_sec:.1f} smp/s | "
                f"VRAM: {vram_mb:.0f} MB | Time: {elapsed:.1f}s"
            )

            record = EpochTelemetry(
                stage_idx=stage.stage_idx,
                epoch=stage_epoch,
                global_epoch=global_epoch_counter,
                train_loss=train_loss,
                train_syntax_loss=train_syn,
                train_gaussian_ord_loss=train_ord,
                train_huber_loss=train_hub,
                train_family_loss=train_fam,
                val_loss=val_loss,
                val_syntax_loss=val_syn,
                val_gaussian_ord_loss=val_ord,
                val_huber_loss=val_hub,
                val_family_loss=val_fam,
                val_family_acc=val_acc,
                learning_rate=optimizer.param_groups[0]["lr"],
                samples_per_sec=samples_sec,
                gpu_memory_used_mb=vram_mb,
                elapsed_seconds=elapsed,
            )
            telemetry_history.append(asdict(record))
            with open(results_dir / "telemetry.json", "w", encoding="utf-8") as tf:
                json.dump(telemetry_history, tf, indent=2)

            latest_cp = TrainingCheckpoint(
                epoch=global_epoch_counter,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
            )
            checkpoint_adapter.save_checkpoint(
                latest_cp, str(checkpoints_dir / "curriculum_v4_latest.pt")
            )

            if val_loss < best_stage_val_loss:
                best_stage_val_loss = val_loss
                stage_best_name = f"curriculum_v4_stage{stage.stage_idx}_best.pt"
                checkpoint_adapter.save_checkpoint(
                    latest_cp, str(checkpoints_dir / stage_best_name)
                )
                logger.log(f"  [+] Saved new Stage {stage.stage_idx} Best: {stage_best_name}")

            if val_loss < best_global_val_loss:
                best_global_val_loss = val_loss
                checkpoint_adapter.save_checkpoint(
                    latest_cp, str(checkpoints_dir / "curriculum_v4_best.pt")
                )
                logger.log(f"  [★] NEW GLOBAL BEST CHECKPOINT: val_loss = {val_loss:.4f}")

            global_epoch_counter += 1

    logger.log("\n================================================================================")
    logger.log(
        f"    TRAINING COMPLETED SUCCESSFULLY! Global Best Val Loss: {best_global_val_loss:.4f}"
    )
    logger.log("================================================================================")
    logger.close()

    if args.auto_shutdown:
        print("[!] Sentinel auto-shutdown triggered. Powering off system to eliminate costs...")
        os.system("sudo poweroff")


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse terminal options for V4 multi-task curriculum training."""
    parser = argparse.ArgumentParser(
        description="V4 Multimodal Multi-Task Curriculum Training Orchestrator."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/sharded_v4",
        help="Directory containing manifest_train.json and manifest_val.json.",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="dataset/sharded_v4/vocabulary_v4.json",
        help="Path to serialized TokenVocabulary JSON.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/curriculum_v4",
        help="Directory for checkpoints, logs, and telemetry.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-GPU forward batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch size = batch_size * accum).",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=512,
        help="Transformer model dimension.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum autoregressive sequence length.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Decoder transformer layers.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Multi-head attention heads.",
    )
    parser.add_argument(
        "--dim-ff",
        type=int,
        default=2048,
        help="Feed-forward dimension.",
    )
    parser.add_argument(
        "--num-encoder-blocks",
        type=int,
        default=8,
        help="Convolutional residual blocks in VisionEncoder.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW decoupled weight decay.",
    )
    parser.add_argument(
        "--lambda-coord",
        type=float,
        default=1.0,
        help="Gaussian Ordinal loss weight.",
    )
    parser.add_argument(
        "--lambda-family",
        type=float,
        default=1.5,
        help="Auxiliary family classification loss weight.",
    )
    parser.add_argument(
        "--lambda-spatial",
        type=float,
        default=2.0,
        help="Continuous Huber spatial coordinate loss weight.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Syntax cross-entropy label smoothing epsilon.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.20,
        help="Bandwidth sigma for Gaussian ordinal transition matrix.",
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=0.10,
        help="Huber delta beta threshold.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=6,
        help="Number of shards to hold simultaneously in RAM.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader subprocess workers (0 for in-process memory-mapped streaming).",
    )
    parser.add_argument(
        "--max-samples-per-epoch",
        type=int,
        default=None,
        help="Optional ceiling on samples per epoch for fast smoke testing.",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Stage index to start training from (1, 2, or 3).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master pseudorandom seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (cuda or cpu).",
    )
    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="Trigger poweroff upon successful completion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    execute_v4_curriculum_training(cli_args)
