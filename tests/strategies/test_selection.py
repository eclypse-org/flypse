"""Tests for the selection strategies."""

from __future__ import annotations

import random

import pytest

from flypse.strategies.selection import (
    poisson,
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
