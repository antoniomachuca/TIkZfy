from pathlib import Path

import torch

from scripts.encode_dataset import build_encoded_dataset


def test_build_encoded_dataset_persists_vocabulary_and_sequences(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    train_directory = dataset_directory / "train"
    val_directory = dataset_directory / "val"
    train_directory.mkdir(parents=True)
    val_directory.mkdir(parents=True)

    train_markup = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
    val_markup = r"\begin{tikzpicture}\node at (0,0) {A};\end{tikzpicture}"
    (train_directory / "sample_00000.tex").write_text(train_markup, encoding="utf-8")
    (val_directory / "sample_00000.tex").write_text(val_markup, encoding="utf-8")

    vocabulary_path = tmp_path / "artifacts" / "vocabulary.json"
    shapes = build_encoded_dataset(dataset_directory, vocabulary_path, max_length=16)

    assert vocabulary_path.exists()
    assert shapes == {"train": (1, 16), "val": (1, 16)}
    assert torch.load(dataset_directory / "train_tokens.pt", weights_only=True).shape == (1, 16)
    assert torch.load(dataset_directory / "val_tokens.pt", weights_only=True).shape == (1, 16)
