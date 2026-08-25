"""Spatial-Aware Multi-Tier Training Pipeline with CoordConv and Continuous Coordinate Loss.

Addresses spatial misalignment and teacher-forcing prior collapse via:
    1. CoordConv 2D Cartesian plane injection in VisionEncoder.
    2. SpatialAwareHybridLoss combining token Cross-Entropy with smooth L1 Huber
       loss over continuous coordinate predictions.
    3. Cosine Annealing with Warmup + AdamW weight decay.

References:
    Liu et al., An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.
    Goodfellow et al., Deep Learning — multi-task loss optimization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import cast

import torch
from torch import nn

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.augmentation import augment_batch
from core.ml.loss import (
    SpatialAwareHybridLoss,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
)
from core.ml.model import VisionAutoregressiveModel, resolve_device
from core.models import TrainingCheckpoint


def train_epoch(
    model: VisionAutoregressiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    batch_size: int,
    use_augmentation: bool = True,
    device: torch.device | None = None,
) -> float:
    """Train one teacher-forced epoch with CoordConv and Spatial Huber loss."""
    model.train()
    target_device = device or next(model.parameters()).device
    dataset_size = int(images.shape[0])
    order = torch.randperm(dataset_size, device=images.device)

    shuffled_images = images[order]
    shuffled_tokens = tokens[order]

    total_loss: float = 0.0
    num_batches: int = 0

    step_start: int = 0
    while step_start < dataset_size:
        step_end: int = min(step_start + batch_size, dataset_size)
        batch_imgs = shuffled_images[step_start:step_end].to(target_device)
        batch_toks = shuffled_tokens[step_start:step_end].to(target_device)

        if use_augmentation:
            batch_imgs = augment_batch(batch_imgs, p=0.4)

        decoder_input, targets = build_teacher_forcing_pair(batch_toks)
        optimizer.zero_grad()

        logits = cast(torch.Tensor, model(batch_imgs, decoder_input))
        loss = cast(torch.Tensor, criterion(logits, targets))

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach().item())
        num_batches += 1
        step_start = step_end

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_epoch(
    model: VisionAutoregressiveModel,
    criterion: nn.Module,
    val_images: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device | None = None,
) -> float:
    """Evaluate validation loss under teacher forcing."""
    model.eval()
    target_device = device or next(model.parameters()).device
    dataset_size = int(val_images.shape[0])
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


def run_training(args: argparse.Namespace) -> None:
    """Run full spatial-aligned training loop."""
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    target_device = resolve_device(args.device)
    print(f"[*] Training on target device: {target_device}")

    # Load dataset & vocabulary
    vocab_adapter = JsonVocabularyAdapter()
    vocabulary = vocab_adapter.load_vocabulary(str(data_dir / "vocabulary.json"))
    print(f"[*] Loaded vocabulary with {len(vocabulary.token_to_index)} tokens.")

    train_images: torch.Tensor = torch.load(data_dir / "train_images.pt", weights_only=True)
    train_tokens: torch.Tensor = torch.load(data_dir / "train_tokens.pt", weights_only=True)
    val_images: torch.Tensor = torch.load(data_dir / "val_images.pt", weights_only=True)
    val_tokens: torch.Tensor = torch.load(data_dir / "val_tokens.pt", weights_only=True)
    print(
        f"[*] Train set: {train_images.shape[0]} samples | Val set: {val_images.shape[0]} samples."
    )

    # Initialize Spatial-Aware Model (with CoordConv enabled)
    model = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        num_encoder_blocks=args.num_encoder_blocks,
        use_coord_conv=True,
        max_length=512,
        device=target_device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model initialized with {total_params:,} parameters (CoordConv: Enabled).")

    # Optimizer, Scheduler & Hybrid Spatial Loss
    optimizer = build_adamw_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (int(train_images.shape[0]) // args.batch_size)
    warmup_steps = int(0.1 * total_steps)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    criterion = SpatialAwareHybridLoss(
        vocabulary=vocabulary,
        spatial_weight=args.spatial_weight,
    ).to(target_device)

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "spatial_best_model.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            images=train_images,
            tokens=train_tokens,
            batch_size=args.batch_size,
            use_augmentation=True,
            device=target_device,
        )
        val_loss = evaluate_epoch(
            model=model,
            criterion=criterion,
            val_images=val_images,
            val_tokens=val_tokens,
            batch_size=args.batch_size,
            device=target_device,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_obj = TrainingCheckpoint(
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
            )
            AtomicCheckpointAdapter().save_checkpoint(ckpt_obj, str(best_checkpoint_path))
            print(f"  [+] Saved new best spatial checkpoint (Loss: {val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n[+] Spatial training finished in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")


def parse_arguments() -> argparse.Namespace:
    """Build CLI parser for spatial-aligned training."""
    parser = argparse.ArgumentParser(description="Train Spatial-Aware Image-to-TikZ Model")
    parser.add_argument("--data-dir", type=str, default="dataset/encoded")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=1536)
    parser.add_argument("--num-encoder-blocks", type=int, default=6)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_arguments())
