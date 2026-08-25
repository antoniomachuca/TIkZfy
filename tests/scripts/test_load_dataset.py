"""Validation for the dataset loader script (Paso 1)."""

from pathlib import Path

import torch
import torchvision.io as io

from scripts.load_dataset import build_encoded_dataset


def _write_sample(directory: Path, index: int, markup: str) -> None:
    """Persist a matched (markup, image) pair for one sample."""
    (directory / f"sample_{index:05d}.tex").write_text(markup, encoding="utf-8")
    image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    io.write_png(image, str(directory / f"sample_{index:05d}.png"))


def test_build_encoded_dataset_persists_tensors(tmp_path: Path) -> None:
    train_directory = tmp_path / "train"
    val_directory = tmp_path / "val"
    train_directory.mkdir(parents=True)
    val_directory.mkdir(parents=True)

    _write_sample(train_directory, 0, r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    _write_sample(
        train_directory, 1, r"\begin{tikzpicture}\draw[red] (0,0) circle (1);\end{tikzpicture}"
    )
    _write_sample(val_directory, 0, r"\begin{tikzpicture}\draw (0,0) -- (2,2);\end{tikzpicture}")

    output_directory = tmp_path / "encoded"
    shapes = build_encoded_dataset(
        tmp_path, output_directory, max_length=32, target_height=8, target_width=8
    )

    assert shapes["train"]["samples"] == 2
    assert shapes["val"]["samples"] == 1
    assert (output_directory / "vocabulary.json").exists()
    assert torch.load(output_directory / "train_tokens.pt", weights_only=True).shape == (2, 32)
    assert torch.load(output_directory / "train_images.pt", weights_only=True).shape == (2, 3, 8, 8)
    assert torch.load(output_directory / "val_tokens.pt", weights_only=True).shape == (1, 32)
    assert torch.load(output_directory / "val_images.pt", weights_only=True).shape == (1, 3, 8, 8)
