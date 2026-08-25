"""
Dataset curation primitives: length budgets, deduplication, deterministic splits.

Pure functions over string and ndarray domains; no I/O, no side effects.
Split indices are vectorized via numpy permutations and grouping arithmetic,
with zero scalar iteration over sample contents.

Reference: Golub & Van Loan, Matrix Computations — index arithmetic over
sorted grouped arrays as a vectorized alternative to per-group loops.
"""

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.exceptions import DomainError

# Conservative proxy for the 512-token encoder budget of the autoregressive decoder
MAX_MARKUP_CHARS: int = 4000


def within_length_budget(markup: str, max_chars: int = MAX_MARKUP_CHARS) -> bool:
    """
    Checks whether a markup sequence fits the decoder sequence budget.

    Args:
        markup (str): Candidate TikZ markup.
        max_chars (int): Inclusive character budget.

    Returns:
        bool: True iff the markup length is within budget.

    Temporal complexity: O(1).
    """
    if max_chars <= 0:
        raise DomainError(f"Character budget must be positive. Got {max_chars}.")
    return len(markup) <= max_chars


def markup_fingerprint(markup: str) -> str:
    """
    Computes the SHA-256 content fingerprint of a markup sequence.

    Args:
        markup (str): Input TikZ markup.

    Returns:
        str: Hexadecimal digest.

    Temporal complexity: O(L) where L is the markup length.
    """
    return hashlib.sha256(markup.encode("utf-8")).hexdigest()


def deduplicate_markups(markups: list[str]) -> list[str]:
    """
    Removes duplicate markups while preserving first-occurrence order.

    Args:
        markups (list[str]): Candidate markup sequence.

    Returns:
        list[str]: Order-preserving deduplicated sequence.

    Temporal complexity: O(N) where N is the number of markups.
    """
    return list(dict.fromkeys(markups))


def train_val_split(
    n_samples: int,
    val_ratio: float,
    seed: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Computes a deterministic train/validation index partition.

    Args:
        n_samples (int): Total sample count. Must be >= 2.
        val_ratio (float): Validation fraction in (0, 1).
        seed (int): Seed fixing the permutation.

    Returns:
        tuple[NDArray[Any], NDArray[Any]]: Sorted (train_indices, val_indices),
        each of shape (n_train,) and (n_val,).

    Raises:
        DomainError: On invalid counts or ratio bounds.

    Temporal complexity: O(N log N) from the index sort, O(N) extra space.
    """
    if n_samples < 2:
        raise DomainError(f"Sample count must be >= 2. Got {n_samples}.")
    if not 0.0 < val_ratio < 1.0:
        raise DomainError(f"Validation ratio must lie in (0, 1). Got {val_ratio}.")

    rng: np.random.Generator = np.random.default_rng(seed)
    # Shape: (n_samples,)
    permutation: NDArray[Any] = rng.permutation(n_samples)

    n_val: int = min(max(int(round(n_samples * val_ratio)), 1), n_samples - 1)
    train_indices: NDArray[Any] = np.sort(permutation[n_val:])
    val_indices: NDArray[Any] = np.sort(permutation[:n_val])
    return train_indices, val_indices


def stratified_train_val_split(
    labels: NDArray[Any],
    val_ratio: float,
    seed: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Computes a deterministic stratified train/validation partition.

    Every label stratum contributes approximately `val_ratio` of its mass to
    the validation set, guaranteeing family coverage on both sides.

    Args:
        labels (NDArray[Any]): Non-negative integer stratum per sample.
            Shape: (n_samples,)
        val_ratio (float): Validation fraction in (0, 1).
        seed (int): Seed fixing the within-stratum ordering.

    Returns:
        tuple[NDArray[Any], NDArray[Any]]: Sorted (train_indices, val_indices).

    Raises:
        DomainError: On empty input, negative labels, or invalid ratio.

    Temporal complexity: O(N log N) from the lexicographical sort.
    """
    if labels.ndim != 1 or labels.shape[0] < 2:
        raise DomainError("Labels must be a 1D array with at least 2 samples.")
    if not 0.0 < val_ratio < 1.0:
        raise DomainError(f"Validation ratio must lie in (0, 1). Got {val_ratio}.")
    if int(labels.min()) < 0:
        raise DomainError("Labels must be non-negative integers.")

    n_samples: int = labels.shape[0]
    rng: np.random.Generator = np.random.default_rng(seed)

    # Vectorized within-stratum shuffle: sort by (label, uniform_key)
    # Shape: (n_samples,)
    order: NDArray[Any] = np.lexsort((rng.random(n_samples), labels))
    sorted_labels: NDArray[Any] = labels[order]

    # Per-stratum mass and starting offsets via bincount/cumsum arithmetic
    # Shape: (n_strata,)
    counts: NDArray[Any] = np.bincount(sorted_labels)
    # Shape: (n_strata,)
    stratum_starts: NDArray[Any] = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1])
    )

    # Within-stratum rank of each sorted position, vectorized
    # Shape: (n_samples,)
    within_rank: NDArray[Any] = np.arange(n_samples) - stratum_starts[sorted_labels]
    # Shape: (n_strata,)
    val_counts: NDArray[Any] = np.floor(counts * val_ratio).astype(np.int64)

    # Shape: (n_samples,) boolean mask in sorted order
    is_val_sorted: NDArray[Any] = within_rank < val_counts[sorted_labels]

    val_indices: NDArray[Any] = np.sort(order[is_val_sorted])
    train_indices: NDArray[Any] = np.sort(order[~is_val_sorted])
    return train_indices, val_indices
