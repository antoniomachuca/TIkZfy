"""Re-train on mixed Tier 1 + Tier 2 data and compare against the baseline (Paso 6).

Merges the Tier 1 and Tier 2 training tensors, re-trains with the baseline
hyperparameters, evaluates on all three tiers, and persists a comparison of the
mixed model against the Tier 1 baseline into ``results/mixed_training_evaluation.json``.

References:
    Goodfellow et al., Deep Learning — data diversification and its effect on
        generalization error (§5.2).
    Papineni et al., BLEU; Levenshtein, Binary Codes; Wang et al., SSIM.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import torch
from torch import nn

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.math.tokenization import batch_encode
from core.ml.checkpoint import snapshot_checkpoint
from core.ml.loss import (
    TeacherForcingCrossEntropy,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
)
from core.ml.model import VisionAutoregressiveModel
from core.ml.trainer import iter_batch_bounds, train_one_epoch
from core.models import TokenVocabulary
from scripts.evaluate_multi_tier import (
    evaluate_tier,
    generalization_gap,
    load_model,
    load_tier_tensors,
)
from scripts.load_dataset import load_image_batch, load_markup_corpus
from scripts.train_baseline import build_model, evaluate_loss, save_loss_curve


def load_mixed_train_tensors(
    encoded_dir: Path,
    tier2_dir: Path,
    vocabulary: TokenVocabulary,
    max_length: int,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge Tier 1 and Tier 2 training tensors, returning train and Tier 1 val."""
    tier1_train_path: Path = encoded_dir / "train_images.pt"
    tier1_token_path: Path = encoded_dir / "train_tokens.pt"
    if tier1_train_path.exists() and tier1_token_path.exists():
        tier1_train_images = torch.load(tier1_train_path, weights_only=True)
        tier1_train_tokens = torch.load(tier1_token_path, weights_only=True)
    else:
        tier1_train_dir: Path = encoded_dir.parent / "processed" / "train"
        print(f"[*] Encoding Tier 1 train tensors on-the-fly from '{tier1_train_dir}'...")
        tier1_corpus = load_markup_corpus(tier1_train_dir)
        tier1_train_tokens = batch_encode(tier1_corpus, vocabulary, max_length)
        tier1_train_images = load_image_batch(tier1_train_dir, target_height, target_width)

    val_images: torch.Tensor = torch.load(
        encoded_dir / "val_images.pt", weights_only=True
    )
    val_tokens: torch.Tensor = torch.load(
        encoded_dir / "val_tokens.pt", weights_only=True
    )

    tier2_train_dir: Path = tier2_dir / "train"
    print(f"[*] Encoding Tier 2 train tensors from '{tier2_train_dir}'...")
    corpus = load_markup_corpus(tier2_train_dir)
    tier2_train_tokens: torch.Tensor = batch_encode(corpus, vocabulary, max_length)
    tier2_train_images: torch.Tensor = load_image_batch(
        tier2_train_dir, target_height, target_width
    )

    train_images: torch.Tensor = torch.cat(
        (tier1_train_images, tier2_train_images), dim=0
    )
    train_tokens: torch.Tensor = torch.cat(
        (tier1_train_tokens, tier2_train_tokens), dim=0
    )
    print(f"[*] Mixed training corpus ready: {train_images.shape[0]} total samples.")
    return train_images, train_tokens, val_images, val_tokens


def train(arguments: argparse.Namespace) -> tuple[dict[str, object], Path]:
    """Run the mixed-corpus training loop and return the summary and checkpoint."""
    encoded_dir: Path = arguments.encoded_dir
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(encoded_dir / "vocabulary.json")
    )

    train_images, train_tokens, val_images, val_tokens = load_mixed_train_tensors(
        encoded_dir,
        arguments.tier2_dir,
        vocabulary,
        arguments.max_length,
        arguments.target_height,
        arguments.target_width,
    )

    torch.manual_seed(arguments.seed)
    model: VisionAutoregressiveModel = build_model(vocabulary, arguments)
    optimizer: torch.optim.AdamW = build_adamw_optimizer(
        model, learning_rate=arguments.lr
    )
    criterion: nn.Module = TeacherForcingCrossEntropy()

    steps_per_epoch: int = len(
        iter_batch_bounds(train_images.shape[0], arguments.batch_size)
    )
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
    final_checkpoint: Path = checkpoint_dir / (
        f"checkpoint_epoch_{arguments.num_epochs:03d}.pt"
    )

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
        print(
            f"epoch {epoch + 1}/{arguments.num_epochs} "
            f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}"
        )
        if (epoch + 1) % arguments.checkpoint_every == 0 or epoch == arguments.num_epochs - 1:
            checkpoint_adapter.save_checkpoint(
                snapshot_checkpoint(model, optimizer, epoch),
                str(final_checkpoint),
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
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    save_loss_curve(epoch_losses, val_losses, output_dir / "loss_curve.png")
    return results, final_checkpoint


async def evaluate_mixed(
    arguments: argparse.Namespace, config: dict[str, object], checkpoint: Path
) -> dict[str, dict[str, Any]]:
    """Evaluate the mixed model on all three tiers and return the tier summary."""
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(arguments.encoded_dir / "vocabulary.json")
    )
    model: VisionAutoregressiveModel = load_model(
        arguments.encoded_dir, checkpoint, config
    )
    processed_dirs: dict[str, Path] = {
        "tier2": arguments.tier2_dir,
        "tier3": arguments.tier3_dir,
    }
    tier_results: dict[str, dict[str, Any]] = {}
    for tier_name in ("tier1", "tier2", "tier3"):
        try:
            images, tokens = load_tier_tensors(
                tier_name,
                arguments.encoded_dir,
                processed_dirs,
                vocabulary,
                arguments.max_length,
                arguments.target_height,
                arguments.target_width,
            )
            print(f"[*] Evaluating mixed model on {tier_name}...")
            tier_results[tier_name] = await evaluate_tier(
                model,
                vocabulary,
                images,
                tokens,
                arguments.max_samples,
                arguments.max_length,
                arguments.workers,
                arguments.target_height,
                arguments.target_width,
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"[!] Skipping {tier_name}: {error}")
    return tier_results


def compare_against_baseline(
    arguments: argparse.Namespace, mixed_tiers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Load the Tier 1 baseline and compute the mixed-model delta per tier."""
    multi_path: Path = arguments.results_dir / "multi_tier_evaluation.json"
    baseline_path: Path = arguments.results_dir / "tier1_evaluation.json"
    baseline: dict[str, Any] = {}

    if multi_path.exists():
        with multi_path.open("r", encoding="utf-8") as handle:
            multi_eval: dict[str, Any] = json.load(handle)
        baseline = multi_eval.get("tiers", {}).get("tier1", {})
    elif baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    if not baseline:
        return {"baseline": {}, "deltas": {}}

    metric_keys: tuple[str, ...] = (
        "corpus_bleu",
        "mean_geometric_edit_distance",
        "compilation_rate",
        "mean_ssim",
    )
    deltas: dict[str, dict[str, float]] = {}
    for tier_name, summary in mixed_tiers.items():
        if tier_name == "tier1":
            deltas[tier_name] = {
                key: float(summary.get(key, 0.0)) - float(baseline.get(key, 0.0))
                for key in metric_keys
            }
        else:
            deltas[tier_name] = generalization_gap(baseline, summary)

    return {"baseline": baseline, "deltas": deltas}


def orchestrate(arguments: argparse.Namespace) -> None:
    """Run training, evaluation, and baseline comparison for the mixed model."""
    checkpoint: Path = (
        arguments.output_dir
        / "checkpoints"
        / f"checkpoint_epoch_{arguments.num_epochs:03d}.pt"
    )
    results_path: Path = arguments.output_dir / "training_results.json"

    if arguments.skip_train and checkpoint.exists() and results_path.exists():
        print(f"[*] Skipping training; using existing checkpoint '{checkpoint}'...")
        with results_path.open("r", encoding="utf-8") as handle:
            results: dict[str, Any] = json.load(handle)
    else:
        results, checkpoint = train(arguments)

    config: dict[str, object] = results["config"]  # type: ignore[assignment]

    mixed_tiers: dict[str, dict[str, Any]] = asyncio.run(
        evaluate_mixed(arguments, config, checkpoint)
    )
    comparison: dict[str, Any] = compare_against_baseline(arguments, mixed_tiers)

    output: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "mixed_tiers": mixed_tiers,
        "baseline": comparison["baseline"],
        "deltas": comparison["deltas"],
        "training": {
            "epoch_losses": results["epoch_losses"],
            "val_losses": results["val_losses"],
        },
    }
    output_path: Path = arguments.results_dir / "mixed_training_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for mixed-corpus training."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Re-train the image-to-TikZ model on Tier 1 + Tier 2 data."
    )
    parser.add_argument(
        "--encoded-dir", type=Path, default=repo_root / "dataset" / "encoded"
    )
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "results" / "mixed"
    )
    parser.add_argument(
        "--tier2-dir", type=Path, default=repo_root / "dataset" / "processed_tier2"
    )
    parser.add_argument(
        "--tier3-dir", type=Path, default=repo_root / "dataset" / "processed_tier3"
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=501)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip re-training if final checkpoint already exists and evaluate directly.",
    )
    return parser


def main() -> None:
    """Run the mixed-corpus training entrypoint."""
    orchestrate(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
