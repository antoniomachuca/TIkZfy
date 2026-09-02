import pytest
import torch

from core.math.tokenization import build_vocabulary
from core.ml.generation import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    BeamHypothesis,
    beam_search,
    best_of_n_search,
    build_grammar_mask,
    decode_indices_to_markup,
    greedy_search,
    sample_search,
)
from core.ml.model import VisionAutoregressiveModel
from core.models import BOS_INDEX, EOS_INDEX, ImageTensor, TikzTokens, TokenVocabulary


def _vocabulary() -> TokenVocabulary:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    return build_vocabulary([sample])


def _model(max_length: int = 16) -> VisionAutoregressiveModel:
    torch.manual_seed(7)
    return VisionAutoregressiveModel(
        vocabulary=_vocabulary(),
        input_channels=3,
        model_dimension=32,
        max_length=max_length,
        num_layers=1,
        num_heads=4,
    )


def _image() -> ImageTensor:
    return ImageTensor(torch.randn(1, 3, 32, 32))


def test_default_max_sequence_length_is_512() -> None:
    assert DEFAULT_MAX_SEQUENCE_LENGTH == 512


def test_greedy_search_returns_bounded_bos_sequence() -> None:
    model = _model(max_length=8)
    model.eval()

    indices = greedy_search(model, _image(), max_length=8)

    assert isinstance(indices, tuple)
    assert indices[0] == BOS_INDEX
    assert len(indices) <= 8
    assert all(isinstance(index, int) for index in indices)


def test_greedy_search_decodes_to_valid_markup() -> None:
    model = _model(max_length=8)
    model.eval()

    markup = decode_indices_to_markup(model.vocabulary, greedy_search(model, _image(), 8))

    assert isinstance(markup, TikzTokens)
    assert r"\begin{tikzpicture}" in markup.markup
    assert r"\end{tikzpicture}" in markup.markup


def test_beam_search_returns_ranked_hypotheses() -> None:
    torch.manual_seed(3)
    model = _model(max_length=8)
    model.eval()

    hypotheses = beam_search(model, _image(), beam_width=4, max_length=8)

    assert len(hypotheses) == 4
    assert all(isinstance(hypothesis, BeamHypothesis) for hypothesis in hypotheses)
    scores = [hypothesis.log_probability for hypothesis in hypotheses]
    assert scores == sorted(scores, reverse=True)


def test_beam_width_one_matches_greedy() -> None:
    torch.manual_seed(5)
    model = _model(max_length=8)
    model.eval()
    image = _image()

    greedy_indices = greedy_search(model, image, max_length=8)
    beam_indices = beam_search(model, image, beam_width=1, max_length=8)[0].tokens

    assert beam_indices == greedy_indices


def test_beam_search_returns_eos_terminated_or_truncated_hypotheses() -> None:
    model = _model(max_length=4)
    model.eval()

    hypotheses = beam_search(model, _image(), beam_width=3, max_length=4)

    assert 1 <= len(hypotheses) <= 3
    for hypothesis in hypotheses:
        assert hypothesis.tokens[0] == BOS_INDEX
        assert len(hypothesis.tokens) <= 4


def test_greedy_search_rejects_invalid_max_length() -> None:
    model = _model(max_length=8)

    with pytest.raises(ValueError):
        greedy_search(model, _image(), max_length=1)
    with pytest.raises(ValueError):
        greedy_search(model, _image(), max_length=model.max_length + 1)


def test_beam_search_rejects_invalid_arguments() -> None:
    model = _model(max_length=8)

    with pytest.raises(ValueError):
        beam_search(model, _image(), beam_width=0, max_length=8)
    with pytest.raises(ValueError):
        beam_search(model, _image(), beam_width=2, max_length=1)
    with pytest.raises(ValueError):
        beam_search(model, _image(), beam_width=2, max_length=8, length_penalty=-1.0)


def test_decode_indices_to_markup_rejects_non_tuple_indices() -> None:
    from core.exceptions import TensorTopologyError

    model = _model()

    with pytest.raises(TensorTopologyError):
        decode_indices_to_markup(model.vocabulary, [BOS_INDEX, EOS_INDEX])  # type: ignore[arg-type]


def test_decode_indices_to_markup_axis_environment() -> None:
    sample = TikzTokens(
        markup=r"\begin{axis}\addplot coordinates {(0,0)};\end{axis}",
        packages=("pgfplots",),
    )
    vocab = build_vocabulary([sample])
    tokens = [
        "\\begin{axis}",
        "\\addplot",
        "coordinates",
        "{",
        "(",
        "0",
        ",",
        "0",
        ")",
        "}",
        ";",
        "\\end{axis}",
    ]
    indices = tuple(vocab.token_to_index[t] for t in tokens)

    result = decode_indices_to_markup(vocab, indices)

    assert isinstance(result, TikzTokens)
    assert "\\begin{axis}" in result.markup
    assert "\\end{axis}" in result.markup
    assert "pgfplots" in result.packages


def test_decode_indices_to_markup_tikzcd_environment() -> None:
    sample = TikzTokens(
        markup=r"\begin{tikzcd} A \arrow[r] & B \end{tikzcd}",
        packages=("tikz-cd",),
    )
    vocab = build_vocabulary([sample])
    tokens = ["\\begin{tikzcd}", "A", "\\arrow", "[", "r", "]", "&", "B", "\\end{tikzcd}"]
    indices = tuple(vocab.token_to_index[t] for t in tokens)

    result = decode_indices_to_markup(vocab, indices)

    assert isinstance(result, TikzTokens)
    assert "\\begin{tikzcd}" in result.markup
    assert "\\end{tikzcd}" in result.markup
    assert "tikz-cd" in result.packages


def test_decode_indices_to_markup_recovers_missing_end_delimiter() -> None:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    vocab = build_vocabulary([sample])
    # Truncated without \end{tikzpicture}
    tokens = ["\\begin{tikzpicture}", "\\draw", "(", "0", ",", "0", ")", ";"]
    indices = tuple(vocab.token_to_index[t] for t in tokens)

    result = decode_indices_to_markup(vocab, indices)

    assert isinstance(result, TikzTokens)
    assert "\\begin{tikzpicture}" in result.markup
    assert "\\end{tikzpicture}" in result.markup


def test_build_grammar_mask_initial_step_only_allows_root_opener() -> None:
    vocab = _vocabulary()
    device = torch.device("cpu")
    mask = build_grammar_mask(vocab, [BOS_INDEX], device=device)

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (len(vocab.token_to_index),)
    begin_idx = vocab.token_to_index[r"\begin{tikzpicture}"]
    assert bool(mask[begin_idx].item()) is True
    assert bool(mask[EOS_INDEX].item()) is False


def test_build_grammar_mask_inside_paren_prohibits_semicolon_and_end() -> None:
    vocab = _vocabulary()
    device = torch.device("cpu")
    prefix = [
        BOS_INDEX,
        vocab.token_to_index[r"\begin{tikzpicture}"],
        vocab.token_to_index[r"\draw"],
        vocab.token_to_index["("],
    ]
    mask = build_grammar_mask(vocab, prefix, device=device)

    assert bool(mask[vocab.token_to_index[";"]].item()) is False
    assert bool(mask[vocab.token_to_index[r"\end{tikzpicture}"]].item()) is False
    assert bool(mask[EOS_INDEX].item()) is False


def test_build_grammar_mask_after_end_only_allows_eos() -> None:
    vocab = _vocabulary()
    device = torch.device("cpu")
    prefix = [
        BOS_INDEX,
        vocab.token_to_index[r"\begin{tikzpicture}"],
        vocab.token_to_index[r"\draw"],
        vocab.token_to_index["("],
        vocab.token_to_index["0"],
        vocab.token_to_index[","],
        vocab.token_to_index["0"],
        vocab.token_to_index[")"],
        vocab.token_to_index[";"],
        vocab.token_to_index[r"\end{tikzpicture}"],
    ]
    mask = build_grammar_mask(vocab, prefix, device=device)

    assert bool(mask[EOS_INDEX].item()) is True
    assert mask.sum().item() == 1


def test_greedy_search_with_grammar_constraint() -> None:
    model = _model(max_length=12)
    model.eval()

    indices = greedy_search(model, _image(), max_length=12, grammar_constrained=True)

    assert isinstance(indices, tuple)
    assert indices[0] == BOS_INDEX
    assert all(isinstance(idx, int) for idx in indices)


def test_sample_search_emits_valid_sequence() -> None:
    torch.manual_seed(42)
    model = _model(max_length=12)
    model.eval()

    indices = sample_search(
        model,
        _image(),
        max_length=12,
        temperature=0.8,
        top_p=0.9,
        grammar_constrained=True,
    )

    assert isinstance(indices, tuple)
    assert indices[0] == BOS_INDEX
    assert all(isinstance(idx, int) for idx in indices)


def test_best_of_n_search_generates_expected_candidate_count() -> None:
    torch.manual_seed(99)
    model = _model(max_length=10)
    model.eval()

    hypotheses = best_of_n_search(
        model,
        _image(),
        n_hypotheses=3,
        max_length=10,
        temperature=0.7,
        top_p=0.9,
        grammar_constrained=True,
    )

    assert len(hypotheses) == 3
    for hyp in hypotheses:
        assert isinstance(hyp, tuple)
        assert hyp[0] == BOS_INDEX

