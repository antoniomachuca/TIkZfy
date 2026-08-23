"""End-to-end baseline training script (Paso 2).

Loads the pre-encoded tensors, instantiates a small CPU-friendly
``VisionAutoregressiveModel``, and runs a teacher-forced training loop with
per-epoch validation loss, atomic checkpoints, and matplotlib loss curves.

References:
    Goodfellow et al., Deep Learning — mini-batch SGD (§8.1.3) and teacher
        forcing (§10.2.1).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.ml.checkpoint import snapshot_checkpoint
from core.ml.loss import (
    TeacherForcingCrossEntropy,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
)
from core.ml.model import VisionAutoregressiveModel
from core.ml.trainer import iter_batch_bounds, train_one_epoch
from core.models import TokenVocabulary


def evaluate_loss(
    model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    batch_size: int,
) -> float:
    """Return the mean teacher-forced validation loss (no gradients)."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start, end in iter_batch_bounds(images.shape[0], batch_size):
            decoder_input, targets = build_teacher_forcing_pair(tokens[start:end])
            logits: torch.Tensor = model(images[start:end], decoder_input)
            losses.append(float(criterion(logits, targets).item()))
    return sum(losses) / len(losses)


def build_model(
    vocabulary: TokenVocabulary, arguments: argparse.Namespace
) -> VisionAutoregressiveModel:
    """Instantiate the small image-to-TikZ model from CLI hyperparameters."""
    return VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=arguments.model_dim,
        max_length=arguments.max_length,
        num_layers=arguments.num_layers,
        num_heads=arguments.num_heads,
    )


def save_loss_curve(
    epoch_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Render train/validation loss curves to a PNG artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs: range = range(1, len(epoch_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_losses, marker="o", label="train")
    plt.plot(epochs, val_losses, marker="s", label="val")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.title("Baseline training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def train(arguments: argparse.Namespace) -> None:
    """Run the full training loop and persist checkpoints and results."""
    encoded_dir: Path = arguments.encoded_dir
    train_tokens: torch.Tensor = torch.load(
        encoded_dir / "train_tokens.pt", weights_only=True
    )
    train_images: torch.Tensor = torch.load(
        encoded_dir / "train_images.pt", weights_only=True
    )
    val_tokens: torch.Tensor = torch.load(encoded_dir / "val_tokens.pt", weights_only=True)
    val_images: torch.Tensor = torch.load(encoded_dir / "val_images.pt", weights_only=True)
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(encoded_dir / "vocabulary.json")
    )

    torch.manual_seed(arguments.seed)
    model: VisionAutoregressiveModel = build_model(vocabulary, arguments)
    optimizer: torch.optim.AdamW = build_adamw_optimizer(model, learning_rate=arguments.lr)
    criterion: nn.Module = TeacherForcingCrossEntropy()

    steps_per_epoch: int = len(iter_batch_bounds(train_images.shape[0], arguments.batch_size))
    total_steps: int = arguments.num_epochs * steps_per_epoch
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=max(1, total_steps // 20),
        total_steps=total_steps,
    )

    checkpoint_adapter: AtomicCheckpointAdapter = AtomicCheckpointAdapter()
    output_dir: Path = arguments.output_dir
    checkpoint_dir: Path = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_losses: list[float] = []
    val_losses: list[float] = []
    step_losses: list[float] = []

    for epoch in range(arguments.num_epochs):
        epoch_steps: list[float] = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            images=train_images,
            tokens=train_tokens,
            batch_size=arguments.batch_size,
            shuffle=True,
            seed=arguments.seed + epoch,
        )
        epoch_loss: float = sum(epoch_steps) / len(epoch_steps)
        val_loss: float = evaluate_loss(
            model, criterion, val_images, val_tokens, arguments.batch_size
        )
        epoch_losses.append(epoch_loss)
        val_losses.append(val_loss)
        step_losses.extend(epoch_steps)
        print(
            f"epoch {epoch + 1}/{arguments.num_epochs} "
            f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}"
        )

        if (epoch + 1) % arguments.checkpoint_every == 0 or epoch == arguments.num_epochs - 1:
            checkpoint_adapter.save_checkpoint(
                snapshot_checkpoint(model, optimizer, epoch),
                str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"),
            )

    results: dict[str, object] = {
        "config": {
            "model_dimension": arguments.model_dim,
            "num_layers": arguments.num_layers,
            "num_heads": arguments.num_heads,
            "max_length": arguments.max_length,
            "num_epochs": arguments.num_epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.lr,
            "vocabulary_size": len(vocabulary.token_to_index),
            "train_samples": int(train_tokens.shape[0]),
            "val_samples": int(val_tokens.shape[0]),
        },
        "epoch_losses": epoch_losses,
        "val_losses": val_losses,
        "step_losses": step_losses,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    save_loss_curve(epoch_losses, val_losses, output_dir / "loss_curve.png")
    print(f"Results persisted to '{output_dir}'.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for baseline training."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a small image-to-TikZ baseline model."
    )
    parser.add_argument(
        "--encoded-dir", type=Path, default=repo_root / "dataset" / "encoded"
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run the baseline training entrypoint."""
    train(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
