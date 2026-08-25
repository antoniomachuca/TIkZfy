"""Multi-tier evaluation with generalization-gap quantification (Paso 5).

Loads the Tier 1 checkpoint, decodes greedily, and reports four metrics —
corpus BLEU, mean geometric edit distance, compilation rate, and mean SSIM —
across three difficulty tiers (Tier 1 val, Tier 2 val, Tier 3 test). The
generalization gap ``Delta_OOD`` is the metric degradation from Tier 1 to each
harder tier.

References:
    Papineni et al., BLEU — modified n-gram precision.
    Levenshtein, Binary Codes — edit distance.
    Wang et al., Image Quality Assessment — structural similarity (SSIM).
    Goodfellow et al., Deep Learning — out-of-distribution generalization.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import torch
import torchvision.io as tio

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.packages import BASE_TIKZ_LIBRARIES
from core.exceptions import DomainError
from core.math.spatial import resize_spatial_dimensions
from core.math.tokenization import batch_encode, decode_from_tensor, tokenize_tikz_markup
from core.ml.generation import decode_indices_to_markup, greedy_search
from core.ml.metrics import (
    batch_geometric_graph_edit_distance,
    evaluate_batch,
    structural_similarity,
)
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary
from scripts.load_dataset import load_image_batch, load_markup_corpus

TIER_SPLITS: dict[str, str] = {
    "tier1": "val",
    "tier2": "val",
    "tier3": "test",
}


def load_model(
    encoded_dir: Path, checkpoint_path: Path, config: dict[str, object]
) -> VisionAutoregressiveModel:
    """Reconstruct the model from the persisted vocabulary and checkpoint."""
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(encoded_dir / "vocabulary.json")
    )
    model: VisionAutoregressiveModel = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=cast(int, config["model_dimension"]),
        max_length=cast(int, config["max_length"]),
        num_layers=cast(int, config["num_layers"]),
        num_heads=cast(int, config["num_heads"]),
    )
    checkpoint = AtomicCheckpointAdapter().load_checkpoint(str(checkpoint_path))
    model.load_state_dict(checkpoint.model_state)
    model.eval()
    return model


def decode_reference_tokens(token_row: torch.Tensor, vocabulary: TokenVocabulary) -> list[str]:
    """Map a padded reference index row back onto its token list."""
    markup: TikzTokens = decode_from_tensor(token_row, vocabulary)
    return tokenize_tikz_markup(markup)


def load_tier_tensors(
    tier_name: str,
    encoded_dir: Path,
    processed_dirs: dict[str, Path],
    vocabulary: TokenVocabulary,
    max_length: int,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load or encode the ``(images, tokens)`` tensors for one tier split."""
    if tier_name == "tier1":
        split: str = TIER_SPLITS[tier_name]
        images: torch.Tensor = torch.load(encoded_dir / f"{split}_images.pt", weights_only=True)
        tokens: torch.Tensor = torch.load(encoded_dir / f"{split}_tokens.pt", weights_only=True)
        return images, tokens

    split = TIER_SPLITS[tier_name]
    split_dir: Path = processed_dirs[tier_name] / split
    corpus: list[TikzTokens] = load_markup_corpus(split_dir)
    token_tensor: torch.Tensor = batch_encode(corpus, vocabulary, max_length)
    image_tensor: torch.Tensor = load_image_batch(split_dir, target_height, target_width)
    return image_tensor, token_tensor


def decode_png_bytes(png: bytes, target_height: int, target_width: int) -> torch.Tensor:
    """Decode PNG bytes into a ``(3, H, W)`` float tensor normalized to [0, 1]."""
    raw: torch.Tensor = torch.frombuffer(bytearray(png), dtype=torch.uint8)
    image: torch.Tensor = tio.decode_png(raw).to(dtype=torch.float32) / 255.0
    resized: torch.Tensor = resize_spatial_dimensions(
        ImageTensor(raw_tensor=image.unsqueeze(0)), target_height, target_width
    ).raw_tensor.squeeze(0)
    if resized.shape[0] == 1:
        resized = resized.repeat(3, 1, 1)
    elif resized.shape[0] == 4:
        resized = resized[:3]
    return resized


async def compilation_and_ssim(
    candidate_markups: list[TikzTokens],
    ground_truth_images: list[torch.Tensor],
    workers: int,
    target_height: int,
    target_width: int,
) -> tuple[int, int, float, int]:
    """Compile candidates, return ``(successes, total, mean_ssim, ssim_samples)``."""
    semaphore: asyncio.Semaphore = asyncio.Semaphore(workers)
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(
        engine="pdflatex", tikz_libraries=BASE_TIKZ_LIBRARIES
    )
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()

    async def process(index: int, markup: TikzTokens) -> tuple[bool, float | None]:
        async with semaphore:
            try:
                compilation = await compiler.compile_tikz(markup)
            except DomainError:
                return False, None
            try:
                png: bytes = await rasterizer.rasterize_pdf(compilation.pdf_data)
                predicted: torch.Tensor = decode_png_bytes(png, target_height, target_width)
                ssim: float = structural_similarity(ground_truth_images[index], predicted)
            except DomainError:
                return True, None
            return True, ssim

    results = await asyncio.gather(
        *[process(index, markup) for index, markup in enumerate(candidate_markups)],
        return_exceptions=True,
    )

    successes: int = 0
    ssim_values: list[float] = []
    for result in results:
        if isinstance(result, tuple):
            compiled, ssim = result
            if compiled:
                successes += 1
            if ssim is not None:
                ssim_values.append(ssim)

    total: int = len(candidate_markups)
    mean_ssim: float = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    return successes, total, mean_ssim, len(ssim_values)


async def evaluate_tier(
    model: VisionAutoregressiveModel,
    vocabulary: TokenVocabulary,
    images: torch.Tensor,
    tokens: torch.Tensor,
    max_samples: int,
    max_length: int,
    workers: int,
    target_height: int,
    target_width: int,
) -> dict[str, Any]:
    """Evaluate the model on one tier and return the four-metric summary."""
    sample_count: int = min(int(images.shape[0]), max_samples)
    references: list[list[str]] = []
    candidates: list[list[str]] = []
    candidate_markups: list[TikzTokens] = []
    ground_truth_images: list[torch.Tensor] = []

    for index in range(sample_count):
        image: ImageTensor = ImageTensor(raw_tensor=images[index].unsqueeze(0))
        reference: list[str] = decode_reference_tokens(tokens[index], vocabulary)
        indices: tuple[int, ...] = greedy_search(model, image, max_length=max_length)
        candidate_markup: TikzTokens = decode_indices_to_markup(vocabulary, indices)
        references.append(reference)
        candidates.append(tokenize_tikz_markup(candidate_markup))
        candidate_markups.append(candidate_markup)
        ground_truth_images.append(images[index])

    metrics = evaluate_batch(references, candidates)
    reference_markups: list[TikzTokens] = [
        decode_from_tensor(tokens[idx], vocabulary) for idx in range(sample_count)
    ]
    graph_edit_distances: tuple[float, ...] = batch_geometric_graph_edit_distance(
        reference_markups, candidate_markups
    )
    mean_graph_distance: float = (
        sum(graph_edit_distances) / len(graph_edit_distances) if graph_edit_distances else 0.0
    )

    successes, total, mean_ssim, ssim_samples = await compilation_and_ssim(
        candidate_markups, ground_truth_images, workers, target_height, target_width
    )

    return {
        "samples": sample_count,
        "corpus_bleu": metrics.bleu_score,
        "mean_geometric_edit_distance": metrics.mean_geometric_distance,
        "mean_graph_edit_distance": mean_graph_distance,
        "compilation_rate": successes / total if total else 0.0,
        "compilation": {"successes": successes, "total": total},
        "mean_ssim": mean_ssim,
        "ssim_samples": ssim_samples,
    }


def generalization_gap(baseline: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    """Return per-metric ``baseline - target`` degradation (generalization gap)."""
    metric_keys: tuple[str, ...] = (
        "corpus_bleu",
        "mean_geometric_edit_distance",
        "mean_graph_edit_distance",
        "compilation_rate",
        "mean_ssim",
    )
    return {key: float(baseline.get(key, 0.0)) - float(target.get(key, 0.0)) for key in metric_keys}


def evaluate(arguments: argparse.Namespace) -> None:
    """Run the multi-tier evaluation and persist the comparison table."""
    encoded_dir: Path = arguments.encoded_dir
    results_dir: Path = arguments.results_dir
    with (results_dir / "training_results.json").open("r", encoding="utf-8") as handle:
        training_results: dict[str, object] = json.load(handle)
    config: dict[str, object] = training_results["config"]  # type: ignore[assignment]

    checkpoint_path: Path = arguments.checkpoint
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: '{checkpoint_path}'.")

    model: VisionAutoregressiveModel = load_model(encoded_dir, checkpoint_path, config)
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(encoded_dir / "vocabulary.json")
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
                encoded_dir,
                processed_dirs,
                vocabulary,
                arguments.max_length,
                arguments.target_height,
                arguments.target_width,
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"[!] Skipping {tier_name}: {error}")
            continue
        print(f"[*] Evaluating {tier_name} ({images.shape[0]} samples)...")
        tier_results[tier_name] = asyncio.run(
            evaluate_tier(
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
        )

    baseline: dict[str, Any] = tier_results.get("tier1", {})
    gaps: dict[str, dict[str, float]] = {
        tier_name: generalization_gap(baseline, summary)
        for tier_name, summary in tier_results.items()
        if tier_name != "tier1" and baseline
    }

    output: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "tiers": tier_results,
        "generalization_gap": gaps,
    }
    output_path: Path = results_dir / "multi_tier_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for multi-tier evaluation."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Multi-tier evaluation with generalization-gap quantification."
    )
    parser.add_argument("--encoded-dir", type=Path, default=repo_root / "dataset" / "encoded")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "results" / "checkpoints" / "checkpoint_epoch_020.pt",
    )
    parser.add_argument("--tier2-dir", type=Path, default=repo_root / "dataset" / "processed_tier2")
    parser.add_argument("--tier3-dir", type=Path, default=repo_root / "dataset" / "processed_tier3")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=501)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    """Run the multi-tier evaluation entrypoint."""
    evaluate(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
