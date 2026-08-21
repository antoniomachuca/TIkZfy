import math

import pytest

from core.ml.metrics import (
    DEFAULT_COORDINATE_SCALE,
    EvaluationMetrics,
    batch_geometric_edit_distance,
    corpus_bleu,
    evaluate_batch,
    geometric_edit_distance,
)

REFERENCE: list[str] = ["\\draw", "(0,0)", "--", "(1,1)", ";"]


def _levenshtein(reference: list[str], candidate: list[str]) -> int:
    rows: int = len(reference) + 1
    columns: int = len(candidate) + 1
    matrix: list[list[int]] = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for column in range(columns):
        matrix[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution: int = 0 if reference[row - 1] == candidate[column - 1] else 1
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + substitution,
            )
    return matrix[rows - 1][columns - 1]


def test_corpus_bleu_perfect_match_is_one() -> None:
    tokens: list[list[str]] = [["\\draw", "(0,0)", "--", "(1,1)", ";"]]

    score = corpus_bleu(tokens, tokens, max_order=4)

    assert score == pytest.approx(1.0)


def test_corpus_bleu_zero_overlap_is_zero() -> None:
    references: list[list[str]] = [["\\draw", "--"]]
    candidates: list[list[str]] = [["\\fill", "->"]]

    assert corpus_bleu(references, candidates, max_order=2) == pytest.approx(0.0)


def test_corpus_bleu_applies_brevity_penalty() -> None:
    references: list[list[str]] = [["a", "b", "c", "d", "e"]]
    candidates: list[list[str]] = [["a", "b", "c", "d"]]

    score = corpus_bleu(references, candidates, max_order=1)

    assert score == pytest.approx(math.exp(-0.25))


def test_corpus_bleu_matches_known_n_gram_precision() -> None:
    references: list[list[str]] = [["a", "b", "c", "d", "e"]]
    candidates: list[list[str]] = [["a", "b", "c", "d"]]

    score = corpus_bleu(references, candidates, max_order=2)

    # Precision is 1.0 at both orders; only the brevity penalty scales the score.
    assert score == pytest.approx(math.exp(-0.25))


def test_corpus_bleu_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        corpus_bleu([], [])
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [["a"], ["b"]])
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [["a"]], max_order=0)
    with pytest.raises(TypeError):
        corpus_bleu([["a"]], "a")


def test_geometric_edit_distance_identical_is_zero() -> None:
    assert geometric_edit_distance(REFERENCE, REFERENCE) == pytest.approx(0.0)


def test_geometric_edit_distance_matches_unit_cost_levenshtein() -> None:
    reference: list[str] = ["\\draw", "--", "\\fill", ";"]
    candidate: list[str] = ["\\draw", "->", ";"]

    normalized = geometric_edit_distance(reference, candidate)

    assert normalized == pytest.approx(
        _levenshtein(reference, candidate) / max(len(reference), len(candidate))
    )


def test_geometric_edit_distance_coordinates_cheaper_than_structural() -> None:
    coordinate_shift: list[str] = ["(0.1, 0.0)"]
    structural_change: list[str] = ["\\fill"]

    coordinate_distance = geometric_edit_distance(["(0,0)"], coordinate_shift)
    structural_distance = geometric_edit_distance(["(0,0)"], structural_change)

    assert coordinate_distance < structural_distance
    assert structural_distance == pytest.approx(1.0)


def test_geometric_edit_distance_scales_with_coordinate_separation() -> None:
    near_candidate: list[str] = ["\\draw", "(0.1, 0.0)", "--", "(1.1, 0.9)", ";"]
    far_candidate: list[str] = ["\\draw", "(5,5)", "--", "(-5,-5)", ";"]
    structural_candidate: list[str] = ["\\draw", "\\fill", "--", "(1,1)", ";"]

    near_distance = geometric_edit_distance(REFERENCE, near_candidate)
    structural_distance = geometric_edit_distance(REFERENCE, structural_candidate)
    far_distance = geometric_edit_distance(REFERENCE, far_candidate)

    assert near_distance < structural_distance < far_distance < 1.0


def test_geometric_edit_distance_empty_reference() -> None:
    assert geometric_edit_distance([], ["\\draw"]) == pytest.approx(1.0)
    assert geometric_edit_distance(["\\draw"], []) == pytest.approx(1.0)


def test_geometric_edit_distance_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError):
        geometric_edit_distance(REFERENCE, REFERENCE, coordinate_scale=0.0)
    with pytest.raises(ValueError):
        geometric_edit_distance(REFERENCE, REFERENCE, coordinate_scale=-1.0)


def test_default_coordinate_scale_matches_canvas_diagonal() -> None:
    assert DEFAULT_COORDINATE_SCALE == pytest.approx(10.0 * math.sqrt(2.0))


def test_batch_geometric_edit_distance_returns_per_sample_trace() -> None:
    references: list[list[str]] = [REFERENCE, REFERENCE]
    candidates: list[list[str]] = [REFERENCE, ["\\draw", "(5,5)", "--", "(-5,-5)", ";"]]

    distances = batch_geometric_edit_distance(references, candidates)

    assert len(distances) == 2
    assert distances[0] == pytest.approx(0.0)
    assert 0.0 < distances[1] < 1.0


def test_evaluate_batch_aggregates_metrics() -> None:
    references: list[list[str]] = [REFERENCE, REFERENCE]
    candidates: list[list[str]] = [REFERENCE, ["\\draw", "(5,5)", "--", "(-5,-5)", ";"]]

    metrics = evaluate_batch(references, candidates, max_order=4)

    assert isinstance(metrics, EvaluationMetrics)
    assert 0.0 < metrics.bleu_score < 1.0
    assert len(metrics.per_sample_geometric_distance) == 2
    assert metrics.per_sample_geometric_distance[0] == pytest.approx(0.0)
    assert metrics.per_sample_geometric_distance[1] > 0.0
    assert metrics.mean_geometric_distance == pytest.approx(
        sum(metrics.per_sample_geometric_distance) / 2
    )


def test_evaluate_batch_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate_batch([REFERENCE], [REFERENCE, REFERENCE])


def test_metrics_operate_on_tokenized_markup() -> None:
    from core.math.tokenization import tokenize_tikz_markup
    from core.models import TikzTokens

    reference = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    candidate = TikzTokens(
        markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
    )

    reference_tokens: list[str] = tokenize_tikz_markup(reference)
    candidate_tokens: list[str] = tokenize_tikz_markup(candidate)

    assert geometric_edit_distance(reference_tokens, candidate_tokens) == pytest.approx(0.0)
