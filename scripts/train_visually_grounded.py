"""Production training script for Visually-Grounded Neural Alignment.

Mitigates Transformer posterior collapse and language prior dominance via:
    1. Word Dropout / Token Masking (p = 0.40) during teacher forcing.
    2. CoordConv 2D Cartesian plane injection.
    3. SpatialAwareHybridLoss (joint CrossEntropy + smooth L1 Huber coordinate loss).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import torch
from torch import nn

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.math.augmentation import add_gaussian_noise, jitter_contrast
from core.ml.loss import (
    SpatialAwareHybridLoss,
    apply_word_dropout,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
)
from core.ml.model import VisionAutoregressiveModel, resolve_device
from core.models import TrainingCheckpoint


def apply_photometric_augmentation(images: torch.Tensor, p: float = 0.4) -> torch.Tensor:
    """Apply vectorized photometric noise and contrast jitter with probability p."""
    augmented = images
    if float(torch.rand(1).item()) < p:
        augmented = add_gaussian_noise(augmented, sigma=0.02)
    if float(torch.rand(1).item()) < p:
        augmented = jitter_contrast(augmented, alpha=1.05)
    return augmented


def train_epoch(
    model: VisionAutoregressiveModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    batch_size: int,
    target_device: torch.device,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    use_augmentation: bool = True,
    word_dropout_p: float = 0.40,
) -> float:
    """Execute one training epoch with word dropout and spatial loss."""
    model.train()
    dataset_size: int = int(images.shape[0])
    order = torch.randperm(dataset_size)
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
            batch_imgs = apply_photometric_augmentation(batch_imgs, p=0.4)

        raw_decoder_input, targets = build_teacher_forcing_pair(batch_toks)
        # Apply Word Dropout to force attention to visual encoder tokens
        decoder_input = apply_word_dropout(raw_decoder_input, dropout_probability=word_dropout_p)

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


def run_training(args: argparse.Namespace) -> None:
    """Execute complete grounded training pipeline."""
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

    # Initialize Model with CoordConv
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
    best_checkpoint_path = checkpoints_dir / "grounded_best_model.pt"
    checkpoint_adapter = AtomicCheckpointAdapter()

    print(
        f"[*] Starting {args.epochs} epochs "
        f"(Batch={args.batch_size}, WordDropout={args.word_dropout})..."
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            images=train_images,
            tokens=train_tokens,
            batch_size=args.batch_size,
            target_device=target_device,
            scheduler=scheduler,
            use_augmentation=True,
            word_dropout_p=args.word_dropout,
        )

        val_loss = evaluate_epoch(
            model=model,
            criterion=criterion,
            val_images=val_images,
            val_tokens=val_tokens,
            batch_size=args.batch_size,
            target_device=target_device,
        )

        improved: bool = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            ckpt = TrainingCheckpoint(
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
            )
            checkpoint_adapter.save_checkpoint(ckpt, str(best_checkpoint_path))

        status_flag = "[* BEST]" if improved else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} {status_flag}"
        )

    print(f"[+] Grounded Training Completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[+] Optimal Checkpoint: {best_checkpoint_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for production training."""
    parser = argparse.ArgumentParser(description="Train visually-grounded neural model.")
    parser.add_argument("--data-dir", type=str, default="dataset/encoded")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--word-dropout", type=float, default=0.40)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=1536)
    parser.add_argument("--num-encoder-blocks", type=int, default=6)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_arguments())
