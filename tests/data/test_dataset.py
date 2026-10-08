"""Tests for the FederatedDataset contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flypse.data import DataView, FederatedDataset
from flypse.data.dataset import _normalize_entities

if TYPE_CHECKING:
    from collections.abc import Sequence


class DummyDataset(FederatedDataset):
    """A dataset assigning one source index per entity."""

    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        return {entity_id: [i] for i, entity_id in enumerate(entity_ids)}


def test_normalize_entities_from_count():
    """A positive count becomes identifiers "0", ..., "n-1"."""
    assert _normalize_entities(3) == ["0", "1", "2"]
    assert _normalize_entities(1) == ["0"]


@pytest.mark.parametrize("count", [0, -1, -10])
def test_normalize_entities_rejects_non_positive_count(count):
    """A non-positive count is rejected."""
    with pytest.raises(ValueError, match="positive"):
        _normalize_entities(count)


def test_normalize_entities_from_sequence():
    """A sequence of identifiers is normalized to strings."""
    assert _normalize_entities(["a", "b"]) == ["a", "b"]


def test_normalize_entities_rejects_empty_sequence():
    """An empty roster is rejected."""
    with pytest.raises(ValueError, match="At least one"):
        _normalize_entities([])


def test_normalize_entities_rejects_duplicates():
    """Duplicate identifiers are rejected."""
    with pytest.raises(ValueError, match="unique"):
        _normalize_entities(["a", "b", "a"])


def test_split_reuses_cache_for_same_roster(source):
    """The same roster returns the cached split object."""
    dataset = DummyDataset(source)

    splits = dataset._split(["a", "b"])

    assert splits == {"a": [0], "b": [1]}
    assert dataset._split(["a", "b"]) is splits


def test_split_recomputes_on_new_roster(source):
    """A different roster triggers a new split."""
    dataset = DummyDataset(source)

    splits = dataset._split(["a", "b"])
    new_splits = dataset._split(["a", "b", "c"])

    assert new_splits == {"a": [0], "b": [1], "c": [2]}
    assert new_splits is not splits


def test_split_accepts_entity_count(source):
    """The roster can be given as the number of entities."""
    dataset = DummyDataset(source)

    assert list(dataset._split(2)) == ["0", "1"]


def test_bind_requires_split(source):
    """Binding before splitting is an error."""
    dataset = DummyDataset(source)

    with pytest.raises(RuntimeError, match="not been split"):
        dataset.bind("a")


def test_bind_unknown_entity(source):
    """Binding an entity outside the roster is an error."""
    dataset = DummyDataset(source)
    dataset._split(["a"])

    with pytest.raises(KeyError, match="Unknown entity"):
        dataset.bind("z")


def test_bind_returns_view_without_mutating_dataset(source):
    """Binding returns a per-entity view and leaves the dataset shared."""
    dataset = DummyDataset(source)
    dataset._split(["a", "b"])

    view_a = dataset.bind("a")
    view_b = dataset.bind("b")

    assert isinstance(view_a, DataView)
    assert [view_a[i] for i in range(len(view_a))] == [0]
    assert [view_b[i] for i in range(len(view_b))] == [1]
    # Binding "b" did not rebind the shared dataset: "a" still sees its data.
    assert [view_a[i] for i in range(len(view_a))] == [0]
