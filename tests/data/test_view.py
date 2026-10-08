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
