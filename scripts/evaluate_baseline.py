"""Evaluate a trained baseline: BLEU, geometric edit distance, compilation rate.

Loads the best checkpoint, decodes each validation image greedily (and
optionally with beam search), and reports token-level corpus BLEU, mean
geometric edit distance, and the real TeX compilation rate.

References:
    Papineni et al., BLEU - modified n-gram precision (see core.ml.metrics).
    Levenshtein, Binary Codes - edit distance (see core.ml.metrics).
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

import torch

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.exceptions import DomainError
from core.math.tokenization import decode_from_tensor, tokenize_tikz_markup
from core.ml.generation import beam_search, decode_indices_to_markup, greedy_search
from core.ml.metrics import evaluate_batch
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary


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


async def compilation_rate(candidates: list[TikzTokens]) -> tuple[int, int]:
    """Return (successes, total) of real TeX compilation attempts."""
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")
    successes: int = 0
    for candidate in candidates:
        try:
            await compiler.compile_tikz(candidate)
            successes += 1
        except DomainError:
            pass
    return successes, len(candidates)


def evaluate(arguments: argparse.Namespace) -> None:
    """Run the evaluation pipeline and persist quantitative metrics."""
    encoded_dir: Path = arguments.encoded_dir
    results_dir: Path = arguments.results_dir
    with (results_dir / "training_results.json").open("r", encoding="utf-8") as handle:
        training_results: dict[str, object] = json.load(handle)
    config: dict[str, object] = training_results["config"]  # type: ignore[assignment]

    val_tokens: torch.Tensor = torch.load(encoded_dir / "val_tokens.pt", weights_only=True)
    val_images: torch.Tensor = torch.load(encoded_dir / "val_images.pt", weights_only=True)
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(
        str(encoded_dir / "vocabulary.json")
    )

    checkpoint_path: Path = arguments.checkpoint
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: '{checkpoint_path}'.")
    model: VisionAutoregressiveModel = load_model(encoded_dir, checkpoint_path, config)

    sample_count: int = min(int(val_tokens.shape[0]), arguments.max_samples)
    references: list[list[str]] = []
    candidates: list[list[str]] = []
    candidate_markups: list[TikzTokens] = []

    for index in range(sample_count):
        image: ImageTensor = ImageTensor(raw_tensor=val_images[index].unsqueeze(0))
        reference: list[str] = decode_reference_tokens(val_tokens[index], vocabulary)
        if arguments.use_beam:
            hypotheses = beam_search(model, image, beam_width=3, max_length=arguments.max_length)
            indices: tuple[int, ...] = hypotheses[0].tokens
        else:
            indices = greedy_search(model, image, max_length=arguments.max_length)
        candidate_markup: TikzTokens = decode_indices_to_markup(vocabulary, indices)
        references.append(reference)
        candidates.append(tokenize_tikz_markup(candidate_markup))
        candidate_markups.append(candidate_markup)

    metrics = evaluate_batch(references, candidates)
    successes, total = asyncio.run(compilation_rate(candidate_markups))

    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "samples": sample_count,
        "decoding": "beam_width=3" if arguments.use_beam else "greedy",
        "corpus_bleu": metrics.bleu_score,
        "mean_geometric_edit_distance": metrics.mean_geometric_distance,
        "compilation_rate": successes / total if total else 0.0,
        "compilation": {"successes": successes, "total": total},
    }
    output_path: Path = results_dir / "tier1_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for baseline evaluation."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate a trained image-to-TikZ baseline."
    )
    parser.add_argument(
        "--encoded-dir", type=Path, default=repo_root / "dataset" / "encoded"
    )
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "results" / "checkpoints" / "checkpoint_epoch_020.pt",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=501)
    parser.add_argument("--use-beam", action="store_true")
    return parser


def main() -> None:
    """Run the baseline evaluation entrypoint."""
    evaluate(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
