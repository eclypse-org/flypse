"""Tests for the UniformDataset class."""

from __future__ import annotations

from flypse.data import UniformDataset


def test_uniform_split_contiguous_blocks(source):
    """The source is split into contiguous balanced blocks."""
    dataset = UniformDataset(source)

    splits = dataset._split(["a", "b", "c"])

    assert splits == {
        "a": [0, 1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    }
    assert dataset.seed is None


def test_uniform_split_distributes_remainder(source):
    """The remainder is given to the first entities."""
    dataset = UniformDataset(source, seed=7)

    splits = dataset._split(4)

    assert [len(splits[entity_id]) for entity_id in ["0", "1", "2", "3"]] == [
        3,
        3,
        2,
        2,
    ]
    assert dataset.seed == 7


def test_uniform_split_partitions_the_source(source):
    """Every source index is assigned to exactly one entity."""
    dataset = UniformDataset(source)

    splits = dataset._split(["a", "b", "c", "d"])

    assigned = sorted(index for indices in splits.values() for index in indices)
    assert assigned == source


def test_uniform_split_empty_source():
    """An empty source yields empty blocks."""
    splits = UniformDataset([])._split(["a", "b"])

    assert splits == {"a": [], "b": []}
