import math

import pytest
import torch

from core.exceptions import VocabularyInvariantError
from core.math.tokenization import (
    CANVAS_MAX,
    CANVAS_MIN,
    COORDINATE_BINS,
    COORDINATE_STEP,
    NUM_COORDINATE_BINS,
    batch_encode,
    build_vocabulary,
    decode_from_tensor,
    encode_to_tensor,
    quantize_coordinate_scalar,
    quantize_coordinate_tuple,
    tokenize_tikz_markup,
)
from core.models.token_vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    TokenVocabulary,
)
from core.models.value_objects import TikzTokens


def test_quantize_coordinate_scalar_precision() -> None:
    assert quantize_coordinate_scalar(1.234) == pytest.approx(1.2)
    assert quantize_coordinate_scalar(-4.567) == pytest.approx(-4.6)
    assert quantize_coordinate_scalar(0.04) == pytest.approx(0.0)
    assert quantize_coordinate_scalar(0.06) == pytest.approx(0.1)


def test_quantize_coordinate_scalar_boundary_clamping() -> None:
    assert quantize_coordinate_scalar(-10.5) == pytest.approx(CANVAS_MIN)
    assert quantize_coordinate_scalar(10.5) == pytest.approx(CANVAS_MAX)
    assert quantize_coordinate_scalar(5.0) == pytest.approx(5.0)
    assert quantize_coordinate_scalar(-5.0) == pytest.approx(-5.0)


def test_quantize_coordinate_scalar_avoids_negative_zero() -> None:
    val = quantize_coordinate_scalar(-0.01)
    assert val == 0.0
    assert math.copysign(1.0, val) == 1.0


def test_quantize_coordinate_scalar_rejects_invalid_inputs() -> None:
    with pytest.raises(VocabularyInvariantError):
        quantize_coordinate_scalar(1.0, min_val=5.0, max_val=-5.0)
    with pytest.raises(VocabularyInvariantError):
        quantize_coordinate_scalar(1.0, step=0.0)
    with pytest.raises(VocabularyInvariantError):
        quantize_coordinate_scalar(1.0, step=-0.1)
    with pytest.raises(TypeError):
        quantize_coordinate_scalar("invalid")  # type: ignore[arg-type]


def test_quantize_coordinate_tuple() -> None:
    point = (1.234, -4.567)
    quantized = quantize_coordinate_tuple(point)
    assert quantized == (1.2, -4.6)


def test_coordinate_bins_cardinality_and_span() -> None:
    assert COORDINATE_STEP == pytest.approx(0.1)
    assert len(COORDINATE_BINS) == NUM_COORDINATE_BINS + 1
    assert COORDINATE_BINS[0] == "-5"
    assert COORDINATE_BINS[-1] == "5"
    assert "0" in COORDINATE_BINS
    assert "1.2" in COORDINATE_BINS
    assert "-4.6" in COORDINATE_BINS


def test_tokenize_splits_latex_commands() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    tokens = tokenize_tikz_markup(sample)
    assert r"\begin{tikzpicture}" in tokens
    assert r"\draw" in tokens
    assert r"\end{tikzpicture}" in tokens


def test_tokenize_splits_coordinates_and_operators() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,1) -- (2,3);\end{tikzpicture}")
    tokens = tokenize_tikz_markup(sample)
    assert "(" in tokens
    assert "0" in tokens
    assert "," in tokens
    assert "1" in tokens
    assert ")" in tokens
    assert "--" in tokens


def test_tokenize_quantizes_continuous_coordinates() -> None:
    sample = TikzTokens(
        markup=r"\begin{tikzpicture}\draw (1.234, -4.567) -- (0.01, 2.99);\end{tikzpicture}"
    )
    tokens = tokenize_tikz_markup(sample, quantize=True)
    assert "1.2" in tokens
    assert "-4.6" in tokens
    assert "0" in tokens
    assert "3" in tokens


def test_build_vocabulary_reserved_indices() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    assert vocab.index_to_token[PAD_INDEX] == "<PAD>"
    assert vocab.index_to_token[BOS_INDEX] == "<BOS>"
    assert vocab.index_to_token[EOS_INDEX] == "<EOS>"
    assert vocab.index_to_token[UNK_INDEX] == "<UNK>"
    assert vocab.token_to_index["<PAD>"] == PAD_INDEX
    assert vocab.token_to_index["<BOS>"] == BOS_INDEX
    assert vocab.token_to_index["<EOS>"] == EOS_INDEX
    assert vocab.token_to_index["<UNK>"] == UNK_INDEX


def test_build_vocabulary_includes_full_spatial_grid() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    vocab = build_vocabulary([sample], include_spatial_grid=True)
    for bin_token in COORDINATE_BINS:
        assert bin_token in vocab.token_to_index


def test_build_vocabulary_bijectivity() -> None:
    sample1 = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    sample2 = TikzTokens(markup=r"\begin{tikzpicture}\node at (0,0) {A};\end{tikzpicture}")
    vocab = build_vocabulary([sample1, sample2])
    assert len(vocab.token_to_index) == len(vocab.index_to_token)
    for token, idx in vocab.token_to_index.items():
        assert vocab.index_to_token[idx] == token


def test_build_vocabulary_deterministic_ordering() -> None:
    sample1 = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    sample2 = TikzTokens(markup=r"\begin{tikzpicture}\node at (0,0) {A};\end{tikzpicture}")
    vocab1 = build_vocabulary([sample1, sample2])
    vocab2 = build_vocabulary([sample1, sample2])
    assert vocab1.token_to_index == vocab2.token_to_index


def test_vocabulary_invariant_rejects_invalid_cardinality() -> None:
    with pytest.raises(VocabularyInvariantError):
        TokenVocabulary(
            token_to_index={"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "a": 4},
            index_to_token={0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"},
        )


def test_encode_produces_long_tensor() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    tensor = encode_to_tensor(sample, vocab, max_length=64)
    assert tensor.dtype == torch.long
    assert tensor.shape == (64,)


def test_encode_prepends_bos_appends_eos() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    tensor = encode_to_tensor(sample, vocab, max_length=32)
    assert tensor[0].item() == BOS_INDEX
    non_pad_indices = [idx.item() for idx in tensor if idx.item() != PAD_INDEX]
    assert non_pad_indices[-1] == EOS_INDEX


def test_encode_unknown_token_maps_to_unk() -> None:
    known = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    unknown = TikzTokens(markup=r"\begin{tikzpicture}\unknowncommand (0,0);\end{tikzpicture}")
    vocab = build_vocabulary([known])
    tensor = encode_to_tensor(unknown, vocab, max_length=32)
    assert UNK_INDEX in tensor.tolist()


def test_encode_pads_short_sequences() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw;\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    tensor = encode_to_tensor(sample, vocab, max_length=16)
    assert tensor[-1].item() == PAD_INDEX


def test_encode_truncates_long_sequences() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    tensor = encode_to_tensor(sample, vocab, max_length=4)
    assert tensor.shape == (4,)


def test_decode_roundtrip_identity() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    tensor = encode_to_tensor(sample, vocab, max_length=64)
    reconstructed = decode_from_tensor(tensor, vocab)
    assert r"\begin{tikzpicture}" in reconstructed.markup
    assert r"\draw" in reconstructed.markup
    assert r"\end{tikzpicture}" in reconstructed.markup


def test_batch_encode_shape_and_dtype() -> None:
    sample1 = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    sample2 = TikzTokens(markup=r"\begin{tikzpicture}\node at (0,0) {A};\end{tikzpicture}")
    vocab = build_vocabulary([sample1, sample2])
    batch_tensor = batch_encode([sample1, sample2], vocab, max_length=32)
    assert batch_tensor.shape == (2, 32)
    assert batch_tensor.dtype == torch.long
