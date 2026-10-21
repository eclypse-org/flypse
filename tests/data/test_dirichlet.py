"""Tests for the DirichletDataset class."""

from __future__ import annotations

import random

import pytest
from eclypse.utils.constants import RND_SEED

from flypse.data import DirichletDataset


def _classes(item: int) -> int:
    """Return the class of an integer item."""
    return item % 3


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_dirichlet_rejects_non_positive_alpha(alpha):
    """A non-positive concentration is rejected."""
    with pytest.raises(ValueError, match="positive"):
        DirichletDataset(list(range(10)), class_fn=_classes, alpha=alpha)


def test_dirichlet_split_is_exact_partition(source):
    """The split covers every source index exactly once."""
    dataset = DirichletDataset(source, class_fn=_classes, alpha=0.5, seed=42)

    splits = dataset._split(["a", "b", "c"])

    assigned = sorted(index for indices in splits.values() for index in indices)
    assert assigned == source


def test_dirichlet_split_is_seed_deterministic(source):
    """The same seed yields the same split."""
    first = DirichletDataset(source, class_fn=_classes, alpha=0.5, seed=42)
    second = DirichletDataset(source, class_fn=_classes, alpha=0.5, seed=42)

    assert first._split(["a", "b", "c"]) == second._split(["a", "b", "c"])


def test_dirichlet_uses_class_fn_on_source_items():
    """The class function is applied to the items yielded by the source."""
    items = [("cat", 0), ("dog", 1), ("cat", 2), ("dog", 3)]
    dataset = DirichletDataset(items, class_fn=lambda item: item[0], seed=0)

    splits = dataset._split(["a", "b"])

    assigned = sorted(index for indices in splits.values() for index in indices)
    assert assigned == [0, 1, 2, 3]


def test_dirichlet_zero_total_falls_back_to_uniform(monkeypatch, source):
    """A zero total of gamma draws falls back to equal proportions."""
    monkeypatch.setattr(random.Random, "gammavariate", lambda self, a, b: 0.0)
    dataset = DirichletDataset(source, class_fn=_classes, alpha=1.0, seed=0)

    splits = dataset._split(["a", "b"])

    assigned = sorted(index for indices in splits.values() for index in indices)
    assert assigned == source


def test_dirichlet_distributes_remainder(monkeypatch):
    """Leftover items go to the entities with the largest remainder."""
    monkeypatch.setattr(random.Random, "gammavariate", lambda self, a, b: 1.0)
    dataset = DirichletDataset(
        list(range(5)),
        class_fn=lambda item: 0,
        alpha=1.0,
        seed=0,
    )

    splits = dataset._split(["a", "b"])

    assert {entity_id: len(indices) for entity_id, indices in splits.items()} == {
        "a": 3,
        "b": 2,
    }
    assert sorted(index for indices in splits.values() for index in indices) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_split_falls_back_to_the_process_seed(monkeypatch, source):
    """Without a seed, the split follows the seed of the process."""
    monkeypatch.setenv(RND_SEED, "7")

    assert DirichletDataset(source, class_fn=_classes, alpha=0.5)._split(
        ["a", "b", "c"]
    ) == DirichletDataset(source, class_fn=_classes, alpha=0.5, seed=7)._split(
        ["a", "b", "c"]
    )
