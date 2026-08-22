"""Load ``(image, markup)`` dataset pairs into ready-to-train PyTorch tensors.

Reads the rendered ``dataset/processed/{train,val}`` splits, tokenizes the TikZ
markup, builds (or reloads) the token vocabulary, encodes sequences to padded
integer indices, and persists the batched tensors under ``dataset/encoded/`` so
the training loop can load them in a single call.

References:
    Goodfellow et al., Deep Learning — batched mini-batch SGD requires fixed
        spatial and sequence shapes (§8.1.3).
"""

import argparse
from pathlib import Path

import torch

from adapters.torchvision_loader import TorchVisionImageLoader
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.math.spatial import normalize_channels, resize_spatial_dimensions
from core.math.tokenization import batch_encode, build_vocabulary
from core.models import ImageTensor, TikzTokens, TokenVocabulary


def _coerce_to_three_channels(image: torch.Tensor) -> torch.Tensor:
    """Map an ``(N, C, H, W)`` uint8 batch onto exactly three RGB channels."""
    channels: int = image.shape[1]
    if channels == 3:
        return image
    if channels == 1:
        return image.repeat(1, 3, 1, 1)
    if channels == 4:
        return image[:, :3, :, :]
    raise ValueError(f"Unsupported channel count {channels}.")


def load_markup_corpus(split_directory: Path) -> list[TikzTokens]:
    """Load TikZ markup files from one dataset split in stable order."""
    markup_paths: list[Path] = sorted(split_directory.glob("*.tex"))
    if not markup_paths:
        raise ValueError(f"No .tex files found in '{split_directory}'.")
    return [TikzTokens(markup=path.read_text(encoding="utf-8")) for path in markup_paths]


def load_image_batch(
    split_directory: Path, target_height: int, target_width: int
) -> torch.Tensor:
    """Load, channel-normalize, resize, and stack a split's image payloads.

    Returns:
        torch.Tensor: Float32 batch ``(N, 3, target_height, target_width)``.
    """
    image_paths: list[Path] = sorted(split_directory.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No .png files found in '{split_directory}'.")

    loader: TorchVisionImageLoader = TorchVisionImageLoader()
    resized_images: list[torch.Tensor] = []
    for path in image_paths:
        rgb: torch.Tensor = _coerce_to_three_channels(loader.load_image(str(path)).raw_tensor)
        batch: ImageTensor = ImageTensor(raw_tensor=rgb)
        normalized: ImageTensor = normalize_channels(batch)
        resized: ImageTensor = resize_spatial_dimensions(
            normalized, target_height, target_width
        )
        resized_images.append(resized.raw_tensor)
    return torch.cat(resized_images, dim=0)


def build_or_load_vocabulary(
    train_directory: Path, vocabulary_path: Path
) -> TokenVocabulary:
    """Build and persist a vocabulary from the train split, or reload it."""
    if vocabulary_path.exists():
        return JsonVocabularyAdapter().load_vocabulary(str(vocabulary_path))
    vocabulary: TokenVocabulary = build_vocabulary(load_markup_corpus(train_directory))
    JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocabulary_path))
    return vocabulary


def encode_and_persist_split(
    split_directory: Path,
    vocabulary: TokenVocabulary,
    output_path: Path,
    max_length: int,
    target_height: int,
    target_width: int,
) -> int:
    """Encode and persist a split's tokens and images; return the token count."""
    corpus: list[TikzTokens] = load_markup_corpus(split_directory)
    token_tensor: torch.Tensor = batch_encode(corpus, vocabulary, max_length)
    image_tensor: torch.Tensor = load_image_batch(
        split_directory, target_height, target_width
    )
    if token_tensor.shape[0] != image_tensor.shape[0]:
        raise ValueError(
            "Image/token sample count mismatch in "
            f"'{split_directory}': {image_tensor.shape[0]} vs {token_tensor.shape[0]}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_tensor, output_path.with_name(f"{output_path.stem}_tokens.pt"))
    torch.save(image_tensor, output_path.with_name(f"{output_path.stem}_images.pt"))
    return int(token_tensor.shape[0])


def build_encoded_dataset(
    dataset_directory: Path,
    output_directory: Path,
    max_length: int,
    target_height: int,
    target_width: int,
) -> dict[str, dict[str, int]]:
    """Persist vocabulary plus encoded image/token tensors for both splits."""
    train_directory: Path = dataset_directory / "train"
    vocabulary_path: Path = output_directory / "vocabulary.json"
    vocabulary: TokenVocabulary = build_or_load_vocabulary(train_directory, vocabulary_path)

    return {
        split_name: {
            "samples": encode_and_persist_split(
                dataset_directory / split_name,
                vocabulary,
                output_directory / split_name,
                max_length,
                target_height,
                target_width,
            ),
            "vocabulary_size": len(vocabulary.token_to_index),
        }
        for split_name in ("train", "val")
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for dataset encoding."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Persist the vocabulary and encoded image/token tensors."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=repo_root / "dataset" / "processed"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "dataset" / "encoded"
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    return parser


def main() -> None:
    """Run the dataset encoding entrypoint."""
    arguments: argparse.Namespace = build_argument_parser().parse_args()
    shapes: dict[str, dict[str, int]] = build_encoded_dataset(
        arguments.dataset_dir,
        arguments.output_dir,
        arguments.max_length,
        arguments.target_height,
        arguments.target_width,
    )
    print(shapes)


if __name__ == "__main__":
    main()
