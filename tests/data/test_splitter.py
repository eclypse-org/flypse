"""Tests for the ViewSplitter and FractionSplitter classes."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from flypse.data import (
    DataView,
    FractionSplitter,
    ViewSplitter,
)


def _view(items: list, indices: Sequence[int] | None = None) -> DataView:
    """Return a view over the given items."""
    return DataView(items, list(range(len(items))) if indices is None else indices)


def _items(view: DataView) -> list:
    """Return the items of a view, in order."""
    return [view[i] for i in range(len(view))]


def _parity(item: int) -> int:
    """Return the class of an integer item."""
    return item % 2


class FirstAndRest(ViewSplitter):
    """A custom splitter: the first position against all the others."""

    def split(self, view: DataView) -> dict[str, Sequence[int]]:
        return {"first": [0], "rest": list(range(1, len(view)))}


def test_apply_returns_views_of_the_parent_view():
    """Parts are views of the split view, which refers to the source."""
    root = _view(list(range(10)), [4, 5, 6, 7])

    parts = FirstAndRest().apply(root)

    assert parts["first"].source is root
    assert _items(parts["first"]) == [4]
    assert _items(parts["rest"]) == [5, 6, 7]


@pytest.mark.parametrize(
    ("parts", "match"),
    [
        ({}, "At least one part"),
        ({"train": 1.2, "test": -0.2}, "positive"),
        ({"train": 0.5, "test": 0.4}, "sum to 1"),
    ],
)
def test_fraction_splitter_rejects_invalid_parts(parts, match):
    """Parts must be non-empty, positive, and sum to 1."""
    with pytest.raises(ValueError, match=match):
        FractionSplitter(parts)


def test_fraction_split_is_exact_partition():
    """Every item of the view goes to exactly one part, sized by fraction."""
    parts = FractionSplitter({"train": 0.7, "eval": 0.1, "test": 0.2}, seed=0).apply(
        _view(list(range(10)))
    )

    assert {name: len(part) for name, part in parts.items()} == {
        "train": 7,
        "eval": 1,
        "test": 2,
    }
    assert sorted(item for part in parts.values() for item in _items(part)) == list(
        range(10)
    )


def test_fraction_split_is_seed_deterministic():
    """The same seed yields the same parts."""
    splitter = FractionSplitter({"train": 0.5, "test": 0.5}, seed=3)
    view = _view(list(range(20)))

    assert splitter.split(view) == FractionSplitter(
        {"train": 0.5, "test": 0.5}, seed=3
    ).split(view)


def test_fraction_split_stratifies_by_class():
    """Each part keeps the class proportions of the view."""
    items = [0] * 8 + [1] * 2  # 80% even, 20% odd
    parts = FractionSplitter(
        {"train": 0.5, "test": 0.5}, stratify=_parity, seed=0
    ).apply(_view(items))

    for part in parts.values():
        assert sorted(_items(part)) == [0, 0, 0, 0, 1]


def test_fraction_split_follows_the_order():
    """Ordered splits give the earliest items to the first parts, in order."""
    timestamps = [5, 1, 9, 3, 7, 2, 8, 0, 6, 4]
    parts = FractionSplitter({"train": 0.6, "test": 0.4}, order=lambda t: t).apply(
        _view(timestamps)
    )

    assert _items(parts["train"]) == [0, 1, 2, 3, 4, 5]
    assert _items(parts["test"]) == [6, 7, 8, 9]


def test_fraction_split_orders_within_each_class():
    """Ordered and stratified splits keep time order inside each class."""
    items = list(range(8))
    parts = FractionSplitter(
        {"train": 0.5, "test": 0.5}, stratify=_parity, order=lambda t: t
    ).apply(_view(items))

    assert _items(parts["train"]) == [0, 1, 2, 3]
    assert _items(parts["test"]) == [4, 5, 6, 7]


def test_fraction_split_allows_empty_parts():
    """A view too small for every part leaves some parts empty."""
    parts = FractionSplitter({"train": 0.7, "eval": 0.1, "test": 0.2}, seed=0).apply(
        _view([1, 2, 3])
    )

    assert sum(len(part) for part in parts.values()) == 3
    assert len(parts["eval"]) == 0


def test_parts_can_be_split_again():
    """A part is a view, so it can be split further."""
    train = FractionSplitter({"train": 0.8, "test": 0.2}, order=lambda t: t).apply(
        _view(list(range(10)))
    )["train"]

    inner = FractionSplitter({"fit": 0.75, "eval": 0.25}, order=lambda t: t).apply(
        train
    )

    assert _items(inner["fit"]) == [0, 1, 2, 3, 4, 5]
    assert _items(inner["eval"]) == [6, 7]
    assert inner["eval"].source is train
