"""Tests for the DataView class."""

from __future__ import annotations

import pytest

from flypse.data import DataView


def test_view_len_and_indexing():
    """The view delegates indexing to the source through its indices."""
    view = DataView(["a", "b", "c", "d"], [3, 1])

    assert len(view) == 2
    assert view[0] == "d"
    assert view[1] == "b"


def test_view_is_empty_without_indices():
    """A view without indices has no items."""
    assert len(DataView([1, 2, 3], [])) == 0


def test_view_out_of_range():
    """Indexing outside the view raises IndexError."""
    view = DataView([10, 20], [0])

    with pytest.raises(IndexError):
        view[5]


def test_view_slice_returns_view():
    """Slicing returns a view of the same source, also on nested views."""
    view = DataView(["a", "b", "c", "d"], [3, 1, 0])
    nested = DataView(view, [2, 0, 1])

    part = view[1:]
    assert isinstance(part, DataView)
    assert part.source is view.source
    assert list(part) == ["b", "a"]
    assert list(nested[:2]) == ["a", "d"]


def test_view_slice_with_step():
    """Slicing with a step selects every step-th item."""
    view = DataView(list(range(10)), [9, 8, 7, 6, 5])

    assert list(view[::2]) == [9, 7, 5]
