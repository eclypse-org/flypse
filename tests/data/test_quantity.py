"""Tests for the QuantitySkewDataset class."""

from __future__ import annotations

import pytest
from eclypse.utils.constants import RND_SEED

from flypse.data import QuantitySkewDataset


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_quantity_rejects_non_positive_alpha(alpha):
    """A non-positive concentration is rejected."""
    with pytest.raises(ValueError, match="positive"):
        QuantitySkewDataset(list(range(10)), alpha=alpha)


def test_quantity_rejects_negative_min_size():
    """A negative minimum size is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        QuantitySkewDataset(list(range(10)), min_size=-1)


def test_quantity_split_is_exact_partition(source):
    """The split covers every source index exactly once."""
    dataset = QuantitySkewDataset(source, alpha=0.5, seed=42)

    splits = dataset._split(["a", "b", "c"])

    assigned = sorted(index for indices in splits.values() for index in indices)
    assert assigned == source


def test_quantity_split_is_seed_deterministic(source):
    """The same seed yields the same split."""
    first = QuantitySkewDataset(source, alpha=0.5, seed=42)
    second = QuantitySkewDataset(source, alpha=0.5, seed=42)

    assert first._split(["a", "b", "c"]) == second._split(["a", "b", "c"])


def test_quantity_respects_min_size():
    """Even a very skewed split gives every entity its minimum."""
    dataset = QuantitySkewDataset(list(range(100)), alpha=0.01, min_size=5, seed=0)

    splits = dataset._split([str(i) for i in range(10)])

    assert min(len(indices) for indices in splits.values()) >= 5


def test_quantity_rejects_a_too_small_source():
    """A source smaller than the minimum sizes cannot be split."""
    dataset = QuantitySkewDataset(list(range(5)), min_size=2)

    with pytest.raises(ValueError, match="cannot give 2"):
        dataset._split(["a", "b", "c"])


def test_quantity_alpha_controls_the_imbalance():
    """Low alpha gives unbalanced sizes, high alpha nearly equal ones."""
    entities = [str(i) for i in range(5)]

    def largest_share(alpha: float) -> float:
        splits = QuantitySkewDataset(list(range(1000)), alpha=alpha, seed=0)._split(
            entities
        )
        return max(len(indices) for indices in splits.values()) / 1000

    assert largest_share(0.1) > 0.5
    assert largest_share(1000.0) < 0.25


def test_split_falls_back_to_the_process_seed(monkeypatch, source):
    """Without a seed, the split follows the seed of the process."""
    monkeypatch.setenv(RND_SEED, "7")

    assert QuantitySkewDataset(source, alpha=0.5)._split(
        ["a", "b", "c"]
    ) == QuantitySkewDataset(source, alpha=0.5, seed=7)._split(["a", "b", "c"])
