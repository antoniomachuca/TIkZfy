import math

import pytest
import torch

from core.ml.metrics import (
    DEFAULT_COORDINATE_SCALE,
    EvaluationMetrics,
    GeometricPrimitive,
    batch_geometric_edit_distance,
    batch_geometric_graph_edit_distance,
    batch_visual_similarity,
    coordinate_error,
    corpus_bleu,
    evaluate_batch,
    geometric_edit_distance,
    geometric_graph_edit_distance,
    structural_similarity,
)
from core.models.value_objects import TikzTokens

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


def test_coordinate_error_extracts_raw_markup_and_reports_points() -> None:
    reference = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
    candidate = r"\begin{tikzpicture}\draw (0.1,0) -- (1.1,1);\end{tikzpicture}"

    result = coordinate_error(reference, candidate)

    assert result.compared_points == 2
    assert result.reference_points == 2
    assert result.candidate_points == 2
    assert result.normalized_error == pytest.approx(0.1 / DEFAULT_COORDINATE_SCALE)


def test_coordinate_error_uses_hungarian_matching_for_unordered_points() -> None:
    reference = r"\begin{tikzpicture}\node at (0,0) {};\node at (4,4) {};\end{tikzpicture}"
    candidate = r"\begin{tikzpicture}\node at (4,4) {};\node at (0,0) {};\end{tikzpicture}"

    ordered = coordinate_error(reference, candidate, order_semantic=True)
    unordered = coordinate_error(reference, candidate, order_semantic=False)

    assert ordered.normalized_error > 0.0
    assert unordered.normalized_error == pytest.approx(0.0)
    assert unordered.matching == "hungarian"


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
    candidate = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")

    reference_tokens: list[str] = tokenize_tikz_markup(reference)
    candidate_tokens: list[str] = tokenize_tikz_markup(candidate)

    assert geometric_edit_distance(reference_tokens, candidate_tokens) == pytest.approx(0.0)


def _fixed_image() -> torch.Tensor:
    """Return a deterministic 64x64 float image with spatial structure."""
    generator: torch.Generator = torch.Generator().manual_seed(7)
    return torch.rand(64, 64, generator=generator)


def test_structural_similarity_identical_is_one() -> None:
    image: torch.Tensor = _fixed_image()

    score = structural_similarity(image, image)

    assert score == pytest.approx(1.0)


def test_structural_similarity_uncorrelated_noise_is_near_zero() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(3)
    noise_a: torch.Tensor = torch.rand(64, 64, generator=generator)
    noise_b: torch.Tensor = torch.rand(64, 64, generator=generator)

    score = structural_similarity(noise_a, noise_b)

    assert score == pytest.approx(0.0, abs=0.05)


def test_structural_similarity_supports_rgb_channels() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(11)
    rgb: torch.Tensor = torch.rand(3, 64, 64, generator=generator)

    assert structural_similarity(rgb, rgb) == pytest.approx(1.0)


def test_structural_similarity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        structural_similarity(torch.rand(64, 64), torch.rand(32, 32))
    with pytest.raises(ValueError):
        structural_similarity(torch.rand(2, 2, 64, 64), torch.rand(2, 2, 64, 64))
    with pytest.raises(ValueError):
        structural_similarity(torch.rand(64, 64), torch.rand(64, 64), data_range=0.0)
    with pytest.raises(ValueError):
        structural_similarity(torch.rand(64, 64), torch.rand(64, 64), window_size=4)


def test_batch_visual_similarity_returns_mean_and_trace() -> None:
    image: torch.Tensor = _fixed_image()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (image, image),
        (image, torch.zeros_like(image)),
    ]

    mean, per_sample = batch_visual_similarity(pairs)

    assert len(per_sample) == 2
    assert per_sample[0] == pytest.approx(1.0)
    assert mean == pytest.approx(sum(per_sample) / 2)


def test_batch_visual_similarity_is_permutation_invariant() -> None:
    image: torch.Tensor = _fixed_image()
    ordered: list[tuple[torch.Tensor, torch.Tensor]] = [
        (image, image),
        (image, torch.zeros_like(image)),
    ]
    permuted: list[tuple[torch.Tensor, torch.Tensor]] = [ordered[1], ordered[0]]

    mean_ordered, trace_ordered = batch_visual_similarity(ordered)
    mean_permuted, trace_permuted = batch_visual_similarity(permuted)

    assert mean_ordered == pytest.approx(mean_permuted)
    assert sorted(trace_ordered) == pytest.approx(sorted(trace_permuted))


def test_batch_visual_similarity_rejects_empty() -> None:
    with pytest.raises(ValueError):
        batch_visual_similarity([])


def test_geometric_primitive_validates_types() -> None:
    primitive = GeometricPrimitive(kind="draw", coordinates=((0.0, 0.0), (1.0, 1.0)))
    assert primitive.kind == "draw"
    assert primitive.coordinates == ((0.0, 0.0), (1.0, 1.0))

    with pytest.raises(TypeError):
        GeometricPrimitive(kind="", coordinates=())
    with pytest.raises(TypeError):
        GeometricPrimitive(kind=123, coordinates=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GeometricPrimitive(kind="draw", coordinates=[(0.0, 0.0)])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GeometricPrimitive(kind="draw", coordinates=((0.0, 0.0, 0.0),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GeometricPrimitive(kind="draw", coordinates=(("0.0", "0.0"),))  # type: ignore[arg-type]


def test_geometric_graph_edit_distance_identical_is_zero() -> None:
    markup = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\node at (2,2) {A};\end{tikzpicture}"
    distance = geometric_graph_edit_distance(markup, markup)
    assert distance == pytest.approx(0.0)


def test_geometric_graph_edit_distance_permutation_invariance() -> None:
    reference = (
        r"\begin{tikzpicture}"
        r"\draw (0,0) -- (1,1);"
        r"\node at (2,2) {A};"
        r"\fill[red] (3,3) circle (1.0);"
        r"\end{tikzpicture}"
    )
    permuted_candidate_a = (
        r"\begin{tikzpicture}"
        r"\fill[red] (3,3) circle (1.0);"
        r"\draw (0,0) -- (1,1);"
        r"\node at (2,2) {A};"
        r"\end{tikzpicture}"
    )
    permuted_candidate_b = (
        r"\begin{tikzpicture}"
        r"\node at (2,2) {A};"
        r"\fill[red] (3,3) circle (1.0);"
        r"\draw (0,0) -- (1,1);"
        r"\end{tikzpicture}"
    )

    distance_a = geometric_graph_edit_distance(reference, permuted_candidate_a)
    distance_b = geometric_graph_edit_distance(reference, permuted_candidate_b)

    assert distance_a == pytest.approx(0.0)
    assert distance_b == pytest.approx(0.0)


def test_geometric_graph_edit_distance_empty_vs_non_empty() -> None:
    non_empty = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
    empty = r"\begin{tikzpicture}\end{tikzpicture}"

    assert geometric_graph_edit_distance(empty, empty) == pytest.approx(0.0)
    assert geometric_graph_edit_distance(non_empty, empty) == pytest.approx(1.0)
    assert geometric_graph_edit_distance(empty, non_empty) == pytest.approx(1.0)


def test_geometric_graph_edit_distance_coordinate_perturbation() -> None:
    reference = r"\draw (0,0) -- (1,1);"
    near_candidate = r"\draw (0.1, 0.0) -- (1.1, 1.0);"
    far_candidate = r"\draw (5.0, 5.0) -- (-5.0, -5.0);"

    near_distance = geometric_graph_edit_distance(reference, near_candidate)
    far_distance = geometric_graph_edit_distance(reference, far_candidate)

    assert 0.0 < near_distance < far_distance <= 1.0


def test_geometric_graph_edit_distance_primitive_type_mismatch() -> None:
    draw_markup = r"\draw (0,0) -- (1,1);"
    fill_markup = r"\fill (0,0) -- (1,1);"

    distance = geometric_graph_edit_distance(draw_markup, fill_markup)
    assert distance == pytest.approx(1.0)


def test_geometric_graph_edit_distance_supports_tikz_tokens() -> None:
    ref_tokens = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")
    cand_tokens = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")

    assert geometric_graph_edit_distance(ref_tokens, cand_tokens) == pytest.approx(0.0)


def test_batch_geometric_graph_edit_distance_returns_per_sample_trace() -> None:
    references = [
        r"\draw (0,0) -- (1,1);",
        r"\draw (0,0) -- (1,1);",
    ]
    candidates = [
        r"\draw (0,0) -- (1,1);",
        r"\draw (5,5) -- (-5,-5);",
    ]

    distances = batch_geometric_graph_edit_distance(references, candidates)

    assert len(distances) == 2
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] > 0.0


def test_batch_geometric_graph_edit_distance_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        batch_geometric_graph_edit_distance([], [])
    with pytest.raises(ValueError):
        batch_geometric_graph_edit_distance([r"\draw (0,0);"], [r"\draw (0,0);", r"\draw (1,1);"])
    with pytest.raises(ValueError):
        geometric_graph_edit_distance(r"\draw (0,0);", r"\draw (0,0);", coordinate_scale=0.0)
    with pytest.raises(ValueError):
        geometric_graph_edit_distance(r"\draw (0,0);", r"\draw (0,0);", coordinate_scale=-2.0)
    with pytest.raises(TypeError):
        geometric_graph_edit_distance(123, r"\draw (0,0);")  # type: ignore[arg-type]
