from pathlib import Path

import pytest

from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.exceptions import DomainError
from core.math.tokenization import build_vocabulary
from core.models.value_objects import TikzTokens


def test_save_load_vocabulary_roundtrip(tmp_path: Path) -> None:
    sample1 = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    sample2 = TikzTokens(markup=r"\begin{tikzpicture}\node at (0,0) {A};\end{tikzpicture}")
    vocab = build_vocabulary([sample1, sample2])

    adapter = JsonVocabularyAdapter()
    file_path = str(tmp_path / "vocab.json")

    adapter.save_vocabulary(vocab, file_path)
    loaded_vocab = adapter.load_vocabulary(file_path)

    assert loaded_vocab.token_to_index == vocab.token_to_index
    assert loaded_vocab.index_to_token == vocab.index_to_token


def test_load_nonexistent_vocabulary_raises_domain_error(tmp_path: Path) -> None:
    adapter = JsonVocabularyAdapter()
    file_path = str(tmp_path / "nonexistent_vocab.json")

    with pytest.raises(DomainError):
        adapter.load_vocabulary(file_path)


def test_load_invalid_json_raises_domain_error(tmp_path: Path) -> None:
    adapter = JsonVocabularyAdapter()
    file_path = tmp_path / "invalid.json"
    file_path.write_text("invalid json payload", encoding="utf-8")

    with pytest.raises(DomainError):
        adapter.load_vocabulary(str(file_path))
