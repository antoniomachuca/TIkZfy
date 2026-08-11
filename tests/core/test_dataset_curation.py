from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from core.dataset import (
    deduplicate_markups,
    markup_fingerprint,
    stratified_train_val_split,
    train_val_split,
    within_length_budget,
)
from core.exceptions import DomainError


def test_within_length_budget() -> None:
    """Verify the budget predicate boundary behavior."""
    assert within_length_budget("x" * 4000)
    assert not within_length_budget("x" * 4001)
    assert within_length_budget("abc", max_chars=3)


def test_invalid_budget_raises() -> None:
    """Verify non-positive budgets are rejected."""
    with pytest.raises(DomainError):
        within_length_budget("abc", max_chars=0)


def test_markup_fingerprint_is_stable() -> None:
    """Verify SHA-256 fingerprint stability and sensitivity."""
    digest_a: str = markup_fingerprint("\\draw (0,0) -- (1,1);")
    digest_b: str = markup_fingerprint("\\draw (0,0) -- (1,1);")
    digest_c: str = markup_fingerprint("\\draw (0,0) -- (1,2);")
    assert digest_a == digest_b
    assert digest_a != digest_c
    assert len(digest_a) == 64


def test_deduplicate_preserves_order() -> None:
    """Verify order-preserving first-occurrence deduplication."""
    assert deduplicate_markups(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert deduplicate_markups([]) == []


def test_train_val_split_sizes_and_disjointness() -> None:
    """Verify partition sizes and empty intersection."""
    train_idx, val_idx = train_val_split(1000, 0.1, seed=42)
    assert train_idx.shape[0] == 900
    assert val_idx.shape[0] == 100
    assert np.intersect1d(train_idx, val_idx).shape[0] == 0
    assert train_idx.max() < 1000 and val_idx.max() < 1000


def test_train_val_split_determinism() -> None:
    """Verify identical seeds reproduce identical partitions."""
    split_a = train_val_split(500, 0.2, seed=7)
    split_b = train_val_split(500, 0.2, seed=7)
    assert np.array_equal(split_a[0], split_b[0])
    assert np.array_equal(split_a[1], split_b[1])


def test_split_rejects_invalid_arguments() -> None:
    """Verify guard clauses on counts and ratio bounds."""
    with pytest.raises(DomainError):
        train_val_split(1, 0.1, seed=0)
    with pytest.raises(DomainError):
        train_val_split(100, 0.0, seed=0)
    with pytest.raises(DomainError):
        train_val_split(100, 1.0, seed=0)


def test_stratified_split_stratum_coverage() -> None:
    """Verify every stratum contributes to both partitions proportionally."""
    # Shape: (800,) — eight strata of 100 samples each
    labels: NDArray[Any] = np.repeat(np.arange(8), 100)
    train_idx, val_idx = stratified_train_val_split(labels, 0.1, seed=42)

    train_labels: NDArray[Any] = np.bincount(labels[train_idx], minlength=8)
    val_labels: NDArray[Any] = np.bincount(labels[val_idx], minlength=8)

    assert np.all(train_labels == 90)
    assert np.all(val_labels == 10)
    assert np.intersect1d(train_idx, val_idx).shape[0] == 0


def test_stratified_split_determinism() -> None:
    """Verify stratified partitions are seed-reproducible."""
    labels: NDArray[Any] = np.repeat(np.arange(4), 50)
    split_a = stratified_train_val_split(labels, 0.25, seed=3)
    split_b = stratified_train_val_split(labels, 0.25, seed=3)
    assert np.array_equal(split_a[0], split_b[0])
    assert np.array_equal(split_a[1], split_b[1])


def test_stratified_split_rejects_invalid_labels() -> None:
    """Verify negative labels and malformed shapes are rejected."""
    with pytest.raises(DomainError):
        stratified_train_val_split(np.array([0, -1, 2]), 0.5, seed=0)
    with pytest.raises(DomainError):
        stratified_train_val_split(np.array([[0, 1], [2, 3]]), 0.5, seed=0)
    with pytest.raises(DomainError):
        stratified_train_val_split(np.array([0]), 0.5, seed=0)
