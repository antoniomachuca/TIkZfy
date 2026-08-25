import numpy as np
import pytest

from core.dataset import (
    DEFAULT_DEPTH_RANGE,
    generate_compositional_batch,
    generate_compositional_sample,
    within_length_budget,
)
from core.exceptions import DomainError
from core.models import TikzTokens


def test_compositional_batch_is_deterministic() -> None:
    """Identical seeds reproduce byte-identical compositional batches."""
    batch_a: list[str] = generate_compositional_batch(32, seed=42)
    batch_b: list[str] = generate_compositional_batch(32, seed=42)
    assert batch_a == batch_b


def test_compositional_batch_distinct_seeds_diverge() -> None:
    """Distinct seeds produce distinct markup."""
    batch_a: list[str] = generate_compositional_batch(16, seed=42)
    batch_b: list[str] = generate_compositional_batch(16, seed=1337)
    assert batch_a != batch_b


def test_compositional_sample_is_valid_tikz_token() -> None:
    """Every generated figure is accepted by the domain value object."""
    rng: np.random.Generator = np.random.default_rng(7)
    samples: list[str] = [
        generate_compositional_sample(rng, depth_range=DEFAULT_DEPTH_RANGE) for _ in range(64)
    ]
    assert all(isinstance(TikzTokens(markup=sample), TikzTokens) for sample in samples)


def test_compositional_sample_respects_length_budget() -> None:
    """Every generated figure fits the decoder length budget."""
    batch: list[str] = generate_compositional_batch(64, seed=99)
    assert all(within_length_budget(markup) for markup in batch)


def test_compositional_sample_scope_count_within_range() -> None:
    """The scope count lands within the requested inclusive depth range."""
    rng: np.random.Generator = np.random.default_rng(11)

    singleton: str = generate_compositional_sample(rng, depth_range=(1, 1))
    body: str = singleton.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert len([line for line in body.splitlines() if line.strip()]) == 1

    full: str = generate_compositional_sample(rng, depth_range=DEFAULT_DEPTH_RANGE)
    body = full.split("\n", 1)[1].rsplit("\n", 1)[0]
    scope_count: int = len([line for line in body.splitlines() if line.strip()])
    assert DEFAULT_DEPTH_RANGE[0] <= scope_count <= DEFAULT_DEPTH_RANGE[1]


def test_compositional_invalid_depth_range_raises() -> None:
    """An invalid depth range is rejected."""
    rng: np.random.Generator = np.random.default_rng(0)
    with pytest.raises(DomainError):
        generate_compositional_sample(rng, depth_range=(0, 5))
    with pytest.raises(DomainError):
        generate_compositional_sample(rng, depth_range=(5, 3))


def test_compositional_non_positive_count_raises() -> None:
    """The batch contract rejects non-positive counts."""
    with pytest.raises(DomainError):
        generate_compositional_batch(0, seed=1)


def test_compositional_batch_is_varied() -> None:
    """A batch contains more than one distinct figure, not degenerate repeats."""
    batch: list[str] = generate_compositional_batch(64, seed=5)
    assert len(set(batch)) > 1
