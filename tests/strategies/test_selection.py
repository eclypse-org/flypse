"""Tests for the selection strategies."""

from __future__ import annotations

import random

import pytest

from flypse.strategies.selection import (
    clustered,
    poisson,
    proportional,
    round_robin,
    uniform,
)


def test_uniform_chooses_k_with_equal_probabilities():
    """k distinct candidates, each selected with probability k / n."""
    selected, probabilities = uniform(["c1", "c2", "c3", "c4"], 2, random.Random(0))

    assert len(set(selected)) == 2
    assert set(selected) <= {"c1", "c2", "c3", "c4"}
    assert probabilities == dict.fromkeys(["c1", "c2", "c3", "c4"], 0.5)


def test_uniform_depends_on_the_seed_not_on_the_order_given():
    """The same seed selects the same candidates, however they are listed."""
    candidates = [f"c{i}" for i in range(10)]

    first, _ = uniform(candidates, 3, random.Random(7))
    second, _ = uniform(reversed(candidates), 3, random.Random(7))

    assert first == second


def test_uniform_chooses_everyone_when_there_are_fewer_than_k():
    """Asking for more than there are selects them all, for sure."""
    selected, probabilities = uniform({"a", "b"}, 5, random.Random(0))

    assert sorted(selected) == ["a", "b"]
    assert probabilities == {"a": 1.0, "b": 1.0}


def test_poisson_chooses_each_candidate_with_probability_q():
    """The fraction selected approaches q."""
    candidates = [f"c{i:04d}" for i in range(4000)]

    selected, probabilities = poisson(candidates, 0.25, random.Random(0))

    assert abs(len(selected) / len(candidates) - 0.25) < 0.03
    assert set(probabilities.values()) == {0.25}


@pytest.mark.parametrize(
    "call",
    [
        lambda: uniform(["a"], -1, random.Random(0)),
        lambda: poisson(["a"], 1.5, random.Random(0)),
    ],
)
def test_selection_refuses_out_of_range_values(call):
    """k is non-negative and q a probability."""
    with pytest.raises(ValueError):
        call()


SIZES = {"a": 50, "b": 30, "c": 15, "d": 5}


def test_proportional_draws_with_replacement():
    """π_i = 1 - (1 - p_i)^k, and a candidate may be drawn more than once."""
    drawn, probabilities = proportional({"a": 3, "b": 1}, 4, random.Random(1))

    assert len(drawn) == 4
    assert probabilities == pytest.approx({"a": 1 - 0.25**4, "b": 1 - 0.75**4})


def test_clustered_draws_one_candidate_per_cluster():
    """A candidate filling a whole cluster is always drawn; π follows the clusters."""
    rng = random.Random(0)

    for _ in range(50):
        drawn, probabilities = clustered(SIZES, 2, rng)
        assert len(drawn) == 2
        assert drawn.count("a") == 1

    assert probabilities == pytest.approx({"a": 1.0, "b": 0.6, "c": 0.3, "d": 0.1})


def test_clustered_keeps_the_expected_draws_of_proportional():
    """Both draw each candidate m p_i times on average."""
    rng = random.Random(0)
    counts = dict.fromkeys(SIZES, 0)
    for _ in range(4000):
        for c in clustered(SIZES, 2, rng)[0]:
            counts[c] += 1

    assert counts["b"] / 4000 == pytest.approx(0.6, abs=0.05)
    assert counts["d"] / 4000 == pytest.approx(0.1, abs=0.03)


def test_round_robin_cycles_through_the_candidates():
    """Turn after turn, k at a time, wrapping around."""
    turns = [round_robin(["e", "d", "c", "b", "a"], 2, t)[0] for t in range(3)]

    assert turns == [["a", "b"], ["c", "d"], ["e", "a"]]
    assert round_robin(["a", "b", "c"], 1, 1)[1] == {"a": 0.0, "b": 1.0, "c": 0.0}
    assert round_robin([], 2, 3) == ([], {})


@pytest.mark.parametrize(
    "call",
    [
        lambda: proportional({"a": 1}, -1, random.Random(0)),
        lambda: proportional({"a": 0}, 1, random.Random(0)),
        lambda: clustered({"a": -1, "b": 2}, 1, random.Random(0)),
        lambda: clustered({"a": 1}, -1, random.Random(0)),
        lambda: round_robin(["a"], 1, -1),
    ],
)
def test_size_based_selection_refuses_invalid_values(call):
    """Draws and turns are non-negative, and so are the sizes, not all zero."""
    with pytest.raises(ValueError):
        call()
