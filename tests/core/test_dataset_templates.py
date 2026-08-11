import numpy as np
import pytest

from core.dataset import (
    FAMILY_NAMES,
    family_index,
    generate_batch,
    generate_sample,
    within_length_budget,
)
from core.exceptions import DomainError
from core.models import TikzTokens


def test_generate_batch_is_deterministic() -> None:
    """Verify identical seeds reproduce byte-identical batches."""
    batch_a: list[str] = generate_batch("polygon", 8, seed=42)
    batch_b: list[str] = generate_batch("polygon", 8, seed=42)
    assert batch_a == batch_b


def test_generate_batch_distinct_seeds_diverge() -> None:
    """Verify distinct seeds produce distinct geometry."""
    batch_a: list[str] = generate_batch("polyline", 8, seed=42)
    batch_b: list[str] = generate_batch("polyline", 8, seed=1337)
    assert batch_a != batch_b


def test_all_families_produce_valid_tikz_tokens() -> None:
    """Verify every family emits markup accepted by the domain value object."""
    rng: np.random.Generator = np.random.default_rng(7)
    samples: list[str] = [generate_sample(family, rng) for family in FAMILY_NAMES]
    tokens: list[TikzTokens] = [TikzTokens(markup=sample) for sample in samples]
    assert len(tokens) == len(FAMILY_NAMES)


def test_all_families_respect_length_budget() -> None:
    """Verify generated markup stays within the decoder sequence budget."""
    for family in FAMILY_NAMES:
        batch: list[str] = generate_batch(family, 32, seed=99)
        assert all(within_length_budget(markup) for markup in batch)


def test_family_count_and_index_bijection() -> None:
    """Verify the family registry exposes exactly eight strata."""
    assert len(FAMILY_NAMES) == 8
    indices: list[int] = [family_index(family) for family in FAMILY_NAMES]
    assert sorted(indices) == list(range(8))


def test_unknown_family_raises_domain_error() -> None:
    """Verify dispatch rejects unregistered family identifiers."""
    rng: np.random.Generator = np.random.default_rng(0)
    with pytest.raises(DomainError):
        generate_sample("hyperbolic_paraboloid", rng)

    with pytest.raises(DomainError):
        family_index("hyperbolic_paraboloid")


def test_non_positive_batch_count_raises() -> None:
    """Verify the batch contract rejects non-positive counts."""
    with pytest.raises(DomainError):
        generate_batch("line_segment", 0, seed=1)
