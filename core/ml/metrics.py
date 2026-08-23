"""Batch evaluation metrics: token BLEU, geometric edit distance, and Hungarian graph edit distance.

Metrics operate over token sequences or raw TikZ markup, keeping evaluation
independent of encoder budget and vocabulary index space.

References:
    Papineni et al., BLEU: a Method for Automatic Evaluation of Machine
        Translation — modified n-gram precision and the brevity penalty.
    Levenshtein, Binary Codes Capable of Correcting Deletions, Insertions and
        Reversals — string edit distance over a unit-cost substitution matrix.
    Golub & Van Loan, Matrix Computations — the edit-distance dynamic program
        is swept column-wise with ``minimum.accumulate`` (a prefix-minimum) so
        each of the O(n) control-flow steps is a vectorized O(m) algebra pass.
    Kuhn, The Hungarian Method for the Assignment Problem / Munkres, Algorithms
        for the Assignment and Transportation Problems — optimal bipartite
        matching for permutation-invariant graph edit distance.
    Tantau, The TikZ and PGF Packages Manual — geometric primitive grammar.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from core.models.value_objects import TikzTokens

# Coordinate literals follow the ``(x, y)`` layout emitted by the procedural
# dataset, whose canvas is the closed square [-5, 5]^2. Its diagonal is
# 10 * sqrt(2); two coordinates separated by that span receive the full unit
# substitution cost, so closer points interpolate the cost linearly in [0, 1].
DEFAULT_COORDINATE_SCALE: float = 10.0 * math.sqrt(2.0)

_COORDINATE_PATTERN: re.Pattern[str] = re.compile(
    r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)"
)


_NUMERIC_PATTERN: re.Pattern[str] = re.compile(r"^-?\d+(?:\.\d+)?$")


def _validate_batch(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
) -> None:
    """Validate a reference/candidate batch pair for length and typing."""
    if not references or not candidates:
        raise ValueError("Reference and candidate batches must be non-empty.")
    if len(references) != len(candidates):
        raise ValueError("Reference and candidate batches must have equal length.")
    for tokens in references:
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise TypeError("Each reference must be a sequence of token strings.")
    for tokens in candidates:
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise TypeError("Each candidate must be a sequence of token strings.")


def _parse_coordinate(token: str) -> tuple[float, float] | None:
    """Return the numeric ``(x, y)`` pair of a coordinate token, else ``None``."""
    match: re.Match[str] | None = _COORDINATE_PATTERN.fullmatch(token)
    coordinates: tuple[float, float] | None = None
    if match is not None:
        coordinates = (float(match.group(1)), float(match.group(2)))
    return coordinates


def _token_points(tokens: Sequence[str]) -> NDArray[np.float64]:
    """Return a ``(L, 2)`` array of coordinate rows, NaN-padded for structural tokens."""
    token_count: int = len(tokens)
    points: NDArray[np.float64] = np.full((token_count, 2), np.nan, dtype=np.float64)
    idx: int = 0
    while idx < token_count:
        token: str = str(tokens[idx])
        match: re.Match[str] | None = _COORDINATE_PATTERN.fullmatch(token)
        if match is not None:
            points[idx] = [float(match.group(1)), float(match.group(2))]
            idx += 1
        elif (
            token == "("
            and idx + 4 < token_count
            and str(tokens[idx + 2]) == ","
            and str(tokens[idx + 4]) == ")"
        ):
            x_match: re.Match[str] | None = _NUMERIC_PATTERN.fullmatch(str(tokens[idx + 1]))
            y_match: re.Match[str] | None = _NUMERIC_PATTERN.fullmatch(str(tokens[idx + 3]))
            if x_match is not None and y_match is not None:
                coord_val: list[float] = [float(tokens[idx + 1]), float(tokens[idx + 3])]
                points[idx] = coord_val
                points[idx + 1] = coord_val
                points[idx + 3] = coord_val
                idx += 5
            else:
                idx += 1
        else:
            idx += 1
    return points


def _substitution_costs(
    reference: Sequence[str],
    candidate: Sequence[str],
    coordinate_scale: float,
) -> NDArray[np.float64]:
    """Build the ``(M, N)`` substitution matrix for coordinate-aware edit distance.

    Identical tokens cost ``0``, structural mismatches cost ``1``, and a
    coordinate pair costs the canvas-normalized Euclidean distance clamped to
    ``[0, 1]``. O(M * N) memory, vectorized via broadcasting.
    """
    reference_length: int = len(reference)
    candidate_length: int = len(candidate)
    costs: NDArray[np.float64] = np.ones(
        (reference_length, candidate_length), dtype=np.float64
    )
    if reference_length == 0 or candidate_length == 0:
        return costs

    reference_tokens: NDArray[np.object_] = np.asarray(list(reference), dtype=object)
    candidate_tokens: NDArray[np.object_] = np.asarray(list(candidate), dtype=object)
    equal_mask: NDArray[np.bool_] = reference_tokens[:, None] == candidate_tokens[None, :]
    costs[equal_mask] = 0.0

    reference_points: NDArray[np.float64] = _token_points(reference)
    candidate_points: NDArray[np.float64] = _token_points(candidate)
    reference_is_coordinate: NDArray[np.bool_] = ~np.isnan(reference_points[:, 0])
    candidate_is_coordinate: NDArray[np.bool_] = ~np.isnan(candidate_points[:, 0])
    coordinate_cells: NDArray[np.bool_] = (
        reference_is_coordinate[:, None] & candidate_is_coordinate[None, :]
    )

    differences: NDArray[np.float64] = reference_points[:, None, :] - candidate_points[None, :, :]
    distances: NDArray[np.float64] = np.linalg.norm(differences, axis=2)
    scaled_distances: NDArray[np.float64] = np.minimum(distances / coordinate_scale, 1.0)
    return np.where(coordinate_cells, scaled_distances, costs)


def _raw_geometric_edit_distance(
    reference: Sequence[str],
    candidate: Sequence[str],
    substitution_costs: NDArray[np.float64],
) -> float:
    """Return the unnormalized coordinate-aware Levenshtein distance.

    The DP matrix is swept column-wise: the deletion and substitution terms are
    evaluated with vectorized algebra and the insertion term is folded in with a
    prefix ``minimum.accumulate``, giving O(n) vectorized steps of O(m) work.
    """
    reference_length: int = len(reference)
    candidate_length: int = len(candidate)
    offsets: NDArray[np.float64] = np.arange(reference_length + 1, dtype=np.float64)
    column: NDArray[np.float64] = offsets.copy()
    for column_index in range(1, candidate_length + 1):
        deletion: NDArray[np.float64] = column + 1.0
        substitution: NDArray[np.float64] = (
            column[:-1] + substitution_costs[:, column_index - 1]
        )
        merged: NDArray[np.float64] = np.empty(reference_length + 1, dtype=np.float64)
        merged[0] = float(column_index)
        merged[1:] = np.minimum(deletion[1:], substitution)
        column = np.minimum.accumulate(merged - offsets) + offsets
    return float(column[reference_length])


def geometric_edit_distance(
    reference: Sequence[str],
    candidate: Sequence[str],
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
) -> float:
    """Return the normalized coordinate-aware edit distance in ``[0, 1]``.

    Identical coordinates contribute a cost proportional to their Euclidean
    separation instead of a binary mismatch, so a slightly perturbed point is
    penalized far less than a wrong structural token.

    Args:
        reference (Sequence[str]): Ground-truth token sequence.
        candidate (Sequence[str]): Generated token sequence.
        coordinate_scale (float): Canvas diagonal in the same units as the
            coordinate literals; used to normalize coordinate substitution costs.

    Returns:
        float: Raw edit distance divided by ``max(len(reference), len(candidate))``.

    Temporal complexity: O(M * N) where M and N are the sequence lengths.
    """
    if coordinate_scale <= 0.0:
        raise ValueError(f"coordinate_scale must be positive. Got {coordinate_scale}.")
    substitution_costs: NDArray[np.float64] = _substitution_costs(
        reference, candidate, coordinate_scale
    )
    raw_distance: float = _raw_geometric_edit_distance(
        reference, candidate, substitution_costs
    )
    return raw_distance / max(len(reference), len(candidate), 1)


def batch_geometric_edit_distance(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
) -> tuple[float, ...]:
    """Return the per-sample normalized geometric edit distance for a batch.

    Temporal complexity: O(sum_i M_i * N_i) over the batch.
    """
    _validate_batch(references, candidates)
    if coordinate_scale <= 0.0:
        raise ValueError(f"coordinate_scale must be positive. Got {coordinate_scale}.")
    return tuple(
        geometric_edit_distance(reference, candidate, coordinate_scale)
        for reference, candidate in zip(references, candidates, strict=True)
    )


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    """Count the ``order``-grams of a token sequence."""
    if len(tokens) < order:
        return Counter()
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _brevity_penalty(reference_length: int, candidate_length: int) -> float:
    """Return the BLEU brevity penalty for total reference/candidate lengths."""
    if candidate_length == 0:
        return 0.0
    if candidate_length > reference_length:
        return 1.0
    return math.exp(1.0 - reference_length / candidate_length)


def corpus_bleu(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    max_order: int = 4,
) -> float:
    """Return corpus-level BLEU in ``[0, 1]`` over token n-grams.

    Each candidate has a single reference, so n-gram counts are clipped to the
    reference count directly. Precision is the geometric mean of modified n-gram
    precisions over orders ``1..max_order``, multiplied by the brevity penalty.

    Args:
        references (Sequence[Sequence[str]]): Ground-truth token sequences.
        candidates (Sequence[Sequence[str]]): Generated token sequences.
        max_order (int): Maximum n-gram order in the precision mean.

    Returns:
        float: BLEU score in ``[0, 1]``.

    Temporal complexity: O(S * max_order * L) where S is the batch size and L
        the mean sequence length.
    """
    _validate_batch(references, candidates)
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1. Got {max_order}.")

    reference_length: int = 0
    candidate_length: int = 0
    clipped_counts: list[int] = [0] * max_order
    total_counts: list[int] = [0] * max_order

    for reference, candidate in zip(references, candidates, strict=True):
        reference_length += len(reference)
        candidate_length += len(candidate)
        for order in range(1, max_order + 1):
            reference_counts: Counter[tuple[str, ...]] = _ngram_counts(reference, order)
            candidate_counts: Counter[tuple[str, ...]] = _ngram_counts(candidate, order)
            clipped_counts[order - 1] += sum(
                min(candidate_counts[ngram], reference_counts[ngram])
                for ngram in candidate_counts
            )
            total_counts[order - 1] += sum(candidate_counts.values())

    precisions: list[float] = [
        clipped / total if total > 0 else 0.0
        for clipped, total in zip(clipped_counts, total_counts, strict=True)
    ]
    positive_precisions: list[float] = [precision for precision in precisions if precision > 0.0]
    if len(positive_precisions) < max_order:
        return 0.0

    log_sum: float = sum(math.log(precision) for precision in positive_precisions)
    geometric_mean: float = math.exp(log_sum / max_order)
    return _brevity_penalty(reference_length, candidate_length) * geometric_mean


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregated batch evaluation: token-level BLEU and geometric edit distance."""

    bleu_score: float
    mean_geometric_distance: float
    per_sample_geometric_distance: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.bleu_score <= 1.0:
            raise ValueError(f"bleu_score must lie in [0, 1]. Got {self.bleu_score}.")
        if not 0.0 <= self.mean_geometric_distance <= 1.0:
            raise ValueError(
                f"mean_geometric_distance must lie in [0, 1]. "
                f"Got {self.mean_geometric_distance}."
            )
        if not self.per_sample_geometric_distance:
            raise ValueError("per_sample_geometric_distance must be non-empty.")
        if not all(
            0.0 <= distance <= 1.0 for distance in self.per_sample_geometric_distance
        ):
            raise ValueError("Every per-sample distance must lie in [0, 1].")


def evaluate_batch(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    max_order: int = 4,
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
) -> EvaluationMetrics:
    """Evaluate a batch with corpus BLEU and per-sample geometric edit distance.

    Args:
        references (Sequence[Sequence[str]]): Ground-truth token sequences.
        candidates (Sequence[Sequence[str]]): Generated token sequences.
        max_order (int): Maximum n-gram order for BLEU.
        coordinate_scale (float): Canvas diagonal for coordinate cost normalization.

    Returns:
        EvaluationMetrics: Corpus BLEU, mean geometric distance and the
            per-sample geometric distance trace.
    """
    corpus_bleu_score: float = corpus_bleu(references, candidates, max_order=max_order)
    per_sample: tuple[float, ...] = batch_geometric_edit_distance(
        references, candidates, coordinate_scale
    )
    mean_distance: float = sum(per_sample) / len(per_sample)
    return EvaluationMetrics(
        bleu_score=corpus_bleu_score,
        mean_geometric_distance=mean_distance,
        per_sample_geometric_distance=per_sample,
    )


# Structural similarity (SSIM) constants: an 11x11 Gaussian window with the
# canonical Wang et al. stability constants on a [0, 1] normalized intensity range.
SSIM_WINDOW_SIZE: int = 11
SSIM_SIGMA: float = 1.5
SSIM_K1: float = 0.01
SSIM_K2: float = 0.03


def _gaussian_window(
    size: int, sigma: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a normalized 2D Gaussian window of shape ``(size, size)``."""
    coords: torch.Tensor = torch.arange(size, dtype=dtype, device=device) - size // 2
    grid: torch.Tensor = coords[:, None] ** 2 + coords[None, :] ** 2
    window: torch.Tensor = torch.exp(-grid / (2.0 * sigma * sigma))
    return window / window.sum()


def _to_batch_channels(image: torch.Tensor) -> torch.Tensor:
    """Coerce a ``(H, W)`` or ``(C, H, W)`` image to ``(1, C, H, W)`` float32."""
    if image.ndim == 2:
        return image.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if image.ndim == 3:
        return image.to(dtype=torch.float32).unsqueeze(0)
    raise ValueError(f"Image must be 2D or 3D. Got {image.ndim}D.")


def structural_similarity(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = SSIM_WINDOW_SIZE,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> float:
    """Return the mean structural similarity index (SSIM) in ``[-1, 1]``.

    Computes luminance, contrast and structure terms over an ``11 x 11``
    Gaussian window with ``k1 = 0.01`` and ``k2 = 0.03``, fully vectorized via
    ``torch.nn.functional.conv2d`` (no per-pixel loops). Accepts grayscale
    ``(H, W)`` or RGB ``(C, H, W)`` tensors and averages over every channel
    and spatial position.

    Args:
        image_a (torch.Tensor): Reference image ``(H, W)`` or ``(C, H, W)``.
        image_b (torch.Tensor): Compared image with identical shape.
        data_range (float): Intensity dynamic range (``1.0`` for normalized images).
        window_size (int): Odd Gaussian window side length.
        k1 (float): Luminance stability constant.
        k2 (float): Contrast stability constant.

    Returns:
        float: Mean SSIM, ``1.0`` for identical images and ``~0`` for
            structurally uncorrelated noise.

    Raises:
        ValueError: On mismatched shapes, unsupported dimensionality, a
            non-positive data range, or an even window size.

    Temporal complexity: O(C * H * W * K^2) where K is the window side length.
    """
    if not isinstance(image_a, torch.Tensor) or not isinstance(image_b, torch.Tensor):
        raise TypeError("Both images must be torch.Tensor instances.")
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Image shapes must match. Got {tuple(image_a.shape)} vs {tuple(image_b.shape)}."
        )
    if image_a.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D. Got {image_a.ndim}D.")
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive. Got {data_range}.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3. Got {window_size}.")

    a: torch.Tensor = _to_batch_channels(image_a)
    b: torch.Tensor = _to_batch_channels(image_b)
    channels: int = a.shape[1]

    window: torch.Tensor = _gaussian_window(
        window_size, SSIM_SIGMA, dtype=a.dtype, device=a.device
    )
    # Shape: (channels, 1, window_size, window_size) so each channel convolves
    # with its own single-channel window via grouped convolution. A valid (zero
    # padding) convolution keeps the statistics fully-supported, so border
    # pixels never see truncated windows.
    kernel: torch.Tensor = window.view(1, 1, window_size, window_size).repeat(
        channels, 1, 1, 1
    )

    c1: float = (k1 * data_range) ** 2
    c2: float = (k2 * data_range) ** 2

    mu_a: torch.Tensor = F.conv2d(a, kernel, groups=channels)
    mu_b: torch.Tensor = F.conv2d(b, kernel, groups=channels)

    mu_a_sq: torch.Tensor = mu_a * mu_a
    mu_b_sq: torch.Tensor = mu_b * mu_b
    mu_ab: torch.Tensor = mu_a * mu_b

    sigma_a_sq: torch.Tensor = F.conv2d(a * a, kernel, groups=channels) - mu_a_sq
    sigma_b_sq: torch.Tensor = F.conv2d(b * b, kernel, groups=channels) - mu_b_sq
    sigma_ab: torch.Tensor = F.conv2d(a * b, kernel, groups=channels) - mu_ab

    numerator: torch.Tensor = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator: torch.Tensor = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map: torch.Tensor = numerator / denominator
    return float(ssim_map.mean().item())


def batch_visual_similarity(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, tuple[float, ...]]:
    """Return the mean and per-sample SSIM over ``(ground_truth, predicted)`` pairs.

    Args:
        pairs (Sequence[tuple[torch.Tensor, torch.Tensor]]): Reference/prediction
            image pairs, each a ``(H, W)`` or ``(C, H, W)`` tensor.

    Returns:
        tuple[float, tuple[float, ...]]: The mean SSIM and the per-sample trace.

    Raises:
        ValueError: If ``pairs`` is empty.

    Temporal complexity: O(sum_i C_i * H_i * W_i * K^2) over the batch.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty.")

    per_sample: tuple[float, ...] = tuple(
        structural_similarity(ground_truth, predicted)
        for ground_truth, predicted in pairs
    )
    return sum(per_sample) / len(per_sample), per_sample


# Regex patterns for lightweight TikZ markup parsing: comments, environment
# wrappers (e.g. \begin{tikzpicture}, \end{tikzpicture}, \pgfplotsset{...}),
# and drawing command keywords.
_LATEX_COMMENT_PATTERN: re.Pattern[str] = re.compile(r"%.*$", re.MULTILINE)
_ENVIRONMENT_STRIP_PATTERN: re.Pattern[str] = re.compile(
    r"\\(?:begin|end)\{[^}]*\}|\\documentclass(?:\[[^\]]*\])?\{[^}]*\}|"
    r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}|\\usetikzlibrary\{[^}]*\}|\\pgfplotsset\{[^}]*\}"
)
_TIKZ_COMMAND_PATTERN: re.Pattern[str] = re.compile(r"\\([a-zA-Z]+)")


@dataclass(frozen=True)
class GeometricPrimitive:
    """Immutable representation of an extracted TikZ geometric drawing primitive.

    Attributes:
        kind (str): Normalized primitive command name (e.g. 'draw', 'fill', 'node', 'path').
        coordinates (tuple[tuple[float, float], ...]): Sequence of 2D Cartesian
            coordinate pairs extracted from the command literal.
    """

    kind: str
    coordinates: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise TypeError("Primitive kind must be a non-empty string.")
        if not isinstance(self.coordinates, tuple):
            raise TypeError("Primitive coordinates must be a tuple.")
        for coord in self.coordinates:
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise TypeError("Each coordinate must be a 2D tuple of length 2.")
            if not all(isinstance(val, (int, float)) for val in coord):
                raise TypeError("Coordinate components must be numeric floats.")


def _extract_markup_text(markup: str | TikzTokens | Sequence[str]) -> str:
    """Extract raw markup text string from str, TikzTokens, or token sequences."""
    if isinstance(markup, str):
        return markup
    if hasattr(markup, "markup") and isinstance(markup.markup, str):
        return markup.markup
    if isinstance(markup, Sequence) and not isinstance(markup, (bytes, bytearray)):
        return " ".join(str(token) for token in markup)
    raise TypeError(f"Expected str, TikzTokens, or Sequence[str]. Got {type(markup)}.")


def _parse_tikz_primitives(
    markup: str | TikzTokens | Sequence[str],
) -> list[GeometricPrimitive]:
    """Parse TikZ drawing statements into structured geometric primitives.

    Strips comments and environment boilerplate, splits statements by semicolon,
    and extracts command keywords with associated Cartesian coordinates.

    Args:
        markup (str | TikzTokens | Sequence[str]): TikZ input representation.

    Returns:
        list[GeometricPrimitive]: Parsed geometric primitives in statement order.

    Temporal complexity: O(L) where L is markup length.
    """
    text: str = _extract_markup_text(markup)
    cleaned: str = _LATEX_COMMENT_PATTERN.sub("", text)
    cleaned = _ENVIRONMENT_STRIP_PATTERN.sub("", cleaned)

    statements: list[str] = cleaned.split(";")
    primitives: list[GeometricPrimitive] = []

    for raw_stmt in statements:
        stmt: str = raw_stmt.strip()
        if stmt:
            cmd_match: re.Match[str] | None = _TIKZ_COMMAND_PATTERN.search(stmt)
            if cmd_match is not None:
                kind: str = cmd_match.group(1).lower()
                coordinates: tuple[tuple[float, float], ...] = tuple(
                    (float(m.group(1)), float(m.group(2)))
                    for m in _COORDINATE_PATTERN.finditer(stmt)
                )
                primitives.append(
                    GeometricPrimitive(kind=kind, coordinates=coordinates)
                )

    return primitives


def _primitive_distance(
    reference: GeometricPrimitive,
    candidate: GeometricPrimitive,
    coordinate_scale: float,
) -> float:
    """Calculate the normalized distance between two geometric primitives in [0, 1].

    - Categorical mismatch in ``kind`` yields unit substitution cost 1.0.
    - Matching kinds compare aligned coordinate sequences: Euclidean distance
      between aligned vertices normalized by ``coordinate_scale`` clamped to [0, 1],
      plus unit penalty for any surplus unmatched coordinates.

    Spatial complexity: O(min(K_r, K_c)) temporary coordinate arrays.
    Temporal complexity: O(min(K_r, K_c)) vectorized Euclidean distance.
    """
    if reference.kind != candidate.kind:
        return 1.0

    ref_coords: tuple[tuple[float, float], ...] = reference.coordinates
    cand_coords: tuple[tuple[float, float], ...] = candidate.coordinates

    ref_len: int = len(ref_coords)
    cand_len: int = len(cand_coords)

    if ref_len == 0 and cand_len == 0:
        return 0.0
    if ref_len == 0 or cand_len == 0:
        return 1.0

    min_len: int = min(ref_len, cand_len)
    max_len: int = max(ref_len, cand_len)

    ref_arr: NDArray[np.float64] = np.asarray(ref_coords[:min_len], dtype=np.float64)
    cand_arr: NDArray[np.float64] = np.asarray(cand_coords[:min_len], dtype=np.float64)

    # Vectorized Euclidean differences across aligned coordinate pairs
    diffs: NDArray[np.float64] = ref_arr - cand_arr  # Shape: (min_len, 2)
    euclidean_dists: NDArray[np.float64] = np.linalg.norm(diffs, axis=1)  # Shape: (min_len,)
    scaled_dists: NDArray[np.float64] = np.minimum(euclidean_dists / coordinate_scale, 1.0)

    total_cost: float = float(np.sum(scaled_dists)) + float(max_len - min_len) * 1.0
    return total_cost / max_len


def _build_primitive_cost_matrix(
    references: Sequence[GeometricPrimitive],
    candidates: Sequence[GeometricPrimitive],
    coordinate_scale: float,
) -> NDArray[np.float64]:
    """Construct the (M, N) bipartite cost matrix between primitive sequences.

    Temporal complexity: O(M * N * K) where K is max coordinate sequence length.
    """
    num_refs: int = len(references)
    num_cands: int = len(candidates)
    cost_matrix: NDArray[np.float64] = np.ones((num_refs, num_cands), dtype=np.float64)

    for ref_idx in range(num_refs):
        for cand_idx in range(num_cands):
            cost_matrix[ref_idx, cand_idx] = _primitive_distance(
                references[ref_idx], candidates[cand_idx], coordinate_scale
            )
    return cost_matrix


def geometric_graph_edit_distance(
    reference_markup: str | TikzTokens,
    candidate_markup: str | TikzTokens,
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
) -> float:
    """Compute permutation-invariant geometric graph edit distance in [0, 1].

    Extracts TikZ geometric primitives (draw, fill, node, path, etc.) and uses
    the Kuhn-Munkres Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    to find the minimum-weight bipartite matching between reference and candidate
    primitives, penalizing unmatched primitives with unit insertion/deletion cost.

    Args:
        reference_markup (str | TikzTokens): Ground truth TikZ markup.
        candidate_markup (str | TikzTokens): Predicted TikZ markup.
        coordinate_scale (float): Canvas diagonal normalization scale.

    Returns:
        float: Normalized graph edit distance in [0, 1].

    Raises:
        ValueError: If coordinate_scale is non-positive.
        TypeError: If markups are not strings, TikzTokens, or token sequences.

    Temporal complexity: O(M * N * K + (M + N)^3) where M, N are primitive counts
        and K is the max coordinate sequence length.
    """
    if coordinate_scale <= 0.0:
        raise ValueError(f"coordinate_scale must be positive. Got {coordinate_scale}.")

    ref_primitives: list[GeometricPrimitive] = _parse_tikz_primitives(reference_markup)
    cand_primitives: list[GeometricPrimitive] = _parse_tikz_primitives(candidate_markup)

    num_refs: int = len(ref_primitives)
    num_cands: int = len(cand_primitives)

    if num_refs == 0 and num_cands == 0:
        return 0.0
    if num_refs == 0 or num_cands == 0:
        return 1.0

    cost_matrix: NDArray[np.float64] = _build_primitive_cost_matrix(
        ref_primitives, cand_primitives, coordinate_scale
    )  # Shape: (num_refs, num_cands)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_cost: float = float(cost_matrix[row_ind, col_ind].sum())

    unmatched_count: int = abs(num_refs - num_cands)
    total_cost: float = matched_cost + float(unmatched_count) * 1.0
    max_primitives: int = max(num_refs, num_cands)

    return total_cost / max_primitives


def batch_geometric_graph_edit_distance(
    references: Sequence[str | TikzTokens],
    candidates: Sequence[str | TikzTokens],
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
) -> tuple[float, ...]:
    """Compute per-sample Hungarian geometric graph edit distance for a batch.

    Args:
        references (Sequence[str | TikzTokens]): Reference markups.
        candidates (Sequence[str | TikzTokens]): Candidate markups.
        coordinate_scale (float): Canvas normalization scale.

    Returns:
        tuple[float, ...]: Tuple of normalized distances in [0, 1].

    Raises:
        ValueError: If batches are empty or lengths mismatch, or scale is invalid.
        TypeError: If elements are of invalid type.

    Temporal complexity: O(sum_i (M_i * N_i * K_i + (M_i + N_i)^3)) over the batch.
    """
    if not references or not candidates:
        raise ValueError("Reference and candidate batches must be non-empty.")
    if len(references) != len(candidates):
        raise ValueError("Reference and candidate batches must have equal length.")
    if coordinate_scale <= 0.0:
        raise ValueError(f"coordinate_scale must be positive. Got {coordinate_scale}.")

    return tuple(
        geometric_graph_edit_distance(ref, cand, coordinate_scale=coordinate_scale)
        for ref, cand in zip(references, candidates, strict=True)
    )
