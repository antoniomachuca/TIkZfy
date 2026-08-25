"""Build and persist the markup vocabulary and encoded dataset sequences."""

import argparse
from pathlib import Path

import torch

from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.math.tokenization import batch_encode, build_vocabulary
from core.models.value_objects import TikzTokens


def load_markup_corpus(split_directory: Path) -> list[TikzTokens]:
    """Load TikZ markup files from one dataset split in stable order."""
    markup_paths: list[Path] = sorted(split_directory.glob("*.tex"))
    if not markup_paths:
        raise ValueError(f"No .tex files found in '{split_directory}'.")

    return [TikzTokens(markup=path.read_text(encoding="utf-8")) for path in markup_paths]


def build_and_persist_vocabulary(train_directory: Path, vocabulary_path: Path) -> int:
    """Build the training vocabulary and return its size."""
    training_corpus: list[TikzTokens] = load_markup_corpus(train_directory)
    vocabulary = build_vocabulary(training_corpus)
    JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocabulary_path))
    return len(vocabulary.token_to_index)


def encode_and_persist_split(
    split_directory: Path,
    vocabulary_path: Path,
    output_path: Path,
    max_length: int,
) -> tuple[int, int]:
    """Encode one split using a persisted vocabulary and save its tensor."""
    corpus: list[TikzTokens] = load_markup_corpus(split_directory)
    vocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocabulary_path))
    encoded_sequences: torch.Tensor = batch_encode(corpus, vocabulary, max_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoded_sequences, output_path)
    return int(encoded_sequences.shape[0]), int(encoded_sequences.shape[1])


def build_encoded_dataset(
    dataset_directory: Path,
    vocabulary_path: Path,
    max_length: int,
) -> dict[str, tuple[int, int]]:
    """Persist the vocabulary and encoded train/validation sequences."""
    train_directory: Path = dataset_directory / "train"
    build_and_persist_vocabulary(train_directory, vocabulary_path)

    split_shapes: dict[str, tuple[int, int]] = {
        split_name: encode_and_persist_split(
            dataset_directory / split_name,
            vocabulary_path,
            dataset_directory / f"{split_name}_tokens.pt",
            max_length,
        )
        for split_name in ("train", "val")
    }
    return split_shapes


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for dataset encoding."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Persist the TikZ vocabulary and encoded token sequences."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--vocabulary-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def main() -> None:
    """Run the dataset encoding entrypoint."""
    arguments: argparse.Namespace = build_argument_parser().parse_args()
    build_encoded_dataset(arguments.dataset_dir, arguments.vocabulary_path, arguments.max_length)


if __name__ == "__main__":
    main()
