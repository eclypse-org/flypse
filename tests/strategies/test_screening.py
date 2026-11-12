"""Tests for the screening strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.strategies.screening import (
    bucketing,
    bulyan,
    cosine_bipartition,
    dnc,
    krum,
    mda,
    monna,
    nnm,
    norm_filter,
)


def _vectors(*rows) -> list[np.ndarray]:
    """Return one float vector per row."""
    return [np.asarray(row, dtype=float) for row in rows]


def test_krum_scores_by_squared_distances():
    """With plain distances 8 and 3 would tie; squared distances favour 3."""
    vectors = _vectors([8.0], [9.0], [3.0], [1.0], [5.0])

    assert krum(vectors, f=1) == [2]


def test_multi_krum_leaves_the_byzantine_contributions_out():
    """The m vectors kept exclude a far-away one."""
    rng = np.random.default_rng(0)
    honest = [rng.normal(0.0, 0.1, 10) for _ in range(6)]

    kept = krum([*honest, np.full(10, 50.0)], f=1, m=5)

    assert len(kept) == 5
    assert 6 not in kept
    assert kept == sorted(kept)


@pytest.mark.parametrize(("n", "f", "m"), [(4, 1, 1), (5, -1, 1), (5, 1, 0), (5, 1, 6)])
def test_krum_checks_its_preconditions(n, f, m):
    """n > 2f + 2, f >= 0 and 1 <= m <= n."""
    with pytest.raises(ValueError):
        krum(_vectors(*([float(i)] for i in range(n))), f=f, m=m)


def _attacked(n_honest: int, seed: int = 0) -> list[np.ndarray]:
    """Return honest vectors near 0, then two far-away ones."""
    rng = np.random.default_rng(seed)
    honest = [rng.normal(0.0, 0.1, 20) for _ in range(n_honest)]
    return [*honest, np.full(20, 5.0), np.full(20, -5.0)]


def test_bulyan_keeps_n_minus_2f_without_the_attackers():
    """θ = n - 2f vectors, chosen by repeated Krum."""
    kept = bulyan(_attacked(9), f=2)

    assert len(kept) == 7
    assert not {9, 10} & set(kept)


def test_bulyan_requires_4f_plus_3_vectors():
    """n >= 4f + 3, and f >= 0."""
    with pytest.raises(ValueError, match="4f"):
        bulyan(_attacked(8), f=2)
    with pytest.raises(ValueError):
        bulyan(_attacked(9), f=-1)


def test_mda_keeps_the_subset_of_smallest_diameter():
    """The far-away vectors widen any subset they are in."""
    assert mda(_attacked(5), f=2) == [0, 1, 2, 3, 4]


def test_norm_filter_keeps_the_short_vectors():
    """The bound is inclusive."""
    assert norm_filter(_vectors([3.0, 4.0], [6.0, 8.0], [0.0, 1.0]), 5.0) == [0, 2]


def test_dnc_drops_the_vectors_along_the_main_spread():
    """The attackers dominate the top singular direction."""
    kept = dnc(_attacked(9), f=2, rng=np.random.default_rng(0), dimensions=10)

    assert kept == list(range(9))


@pytest.mark.parametrize(
    "values",
    [{"f": -1}, {"f": 1, "c": -1.0}, {"f": 1, "iterations": 0}, {"f": 11}],
)
def test_dnc_checks_its_parameters(values):
    """f, c >= 0, at least one iteration, and fewer drops than vectors."""
    with pytest.raises(ValueError, match="DnC"):
        dnc(_attacked(9), rng=np.random.default_rng(0), **values)


def test_cosine_bipartition_separates_opposite_directions():
    """Two clusters pulling in opposite directions."""
    vectors = _vectors([1.0, 0.1], [0.9, 0.0], [-1.0, 0.1], [-0.8, -0.1], [0.0, 0.0])

    assert cosine_bipartition(vectors[:4]) == [[0, 1], [2, 3]]
    assert sorted(map(len, cosine_bipartition(vectors))) == [2, 3]
    assert cosine_bipartition(vectors[:2]) == [[0], [1]]


def test_cosine_bipartition_requires_two_vectors():
    """A single vector cannot be split."""
    with pytest.raises(ValueError, match="two"):
        cosine_bipartition(_vectors([1.0]))


def test_bucketing_shuffles_a_partition_of_the_contributions():
    """Every position once, in buckets of the size, in an order set by the rng."""
    buckets = bucketing(7, 3, np.random.default_rng(0))

    assert sorted(i for b in buckets for i in b) == list(range(7))
    assert [len(b) for b in buckets] == [3, 3, 1]
    assert buckets != bucketing(7, 3, np.random.default_rng(1))
    with pytest.raises(ValueError, match="size"):
        bucketing(3, 0, np.random.default_rng(0))


def test_nnm_groups_each_vector_with_its_nearest():
    """Each group has n - f vectors, the vector itself included."""
    vectors = _vectors([0.0], [0.1], [5.0], [5.2])

    assert nnm(vectors, f=2) == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert monna(vectors, own=3, f=1) == [1, 2, 3]


@pytest.mark.parametrize("call", [lambda v: mda(v, 3), lambda v: monna(v, 0, -1)])
def test_subset_screens_refuse_f_out_of_range(call):
    """f between 0 and n - 1."""
    with pytest.raises(ValueError, match="f must be"):
        call(_vectors([0.0], [1.0], [2.0]))
