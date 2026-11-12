"""Tests for the sampling strategies."""

from __future__ import annotations

import random

import pytest

from flypse.strategies.sampling import (
    poisson,
    uniform,
)


def test_uniform_chooses_k_with_equal_probabilities():
    """k distinct candidates, each chosen with probability k / n."""
    chosen, probabilities = uniform(["c1", "c2", "c3", "c4"], 2, random.Random(0))

    assert len(set(chosen)) == 2
    assert set(chosen) <= {"c1", "c2", "c3", "c4"}
    assert probabilities == dict.fromkeys(["c1", "c2", "c3", "c4"], 0.5)


def test_uniform_depends_on_the_seed_not_on_the_order_given():
    """The same seed chooses the same candidates, however they are listed."""
    candidates = [f"c{i}" for i in range(10)]

    first, _ = uniform(candidates, 3, random.Random(7))
    second, _ = uniform(reversed(candidates), 3, random.Random(7))

    assert first == second


def test_uniform_chooses_everyone_when_there_are_fewer_than_k():
    """Asking for more than there are chooses them all, for sure."""
    chosen, probabilities = uniform({"a", "b"}, 5, random.Random(0))

    assert sorted(chosen) == ["a", "b"]
    assert probabilities == {"a": 1.0, "b": 1.0}


def test_poisson_chooses_each_candidate_with_probability_q():
    """The fraction chosen approaches q."""
    candidates = [f"c{i:04d}" for i in range(4000)]

    chosen, probabilities = poisson(candidates, 0.25, random.Random(0))

    assert abs(len(chosen) / len(candidates) - 0.25) < 0.03
    assert set(probabilities.values()) == {0.25}


@pytest.mark.parametrize(
    "call",
    [
        lambda: uniform(["a"], -1, random.Random(0)),
        lambda: poisson(["a"], 1.5, random.Random(0)),
    ],
)
def test_sampling_refuses_out_of_range_values(call):
    """k is non-negative and q a probability."""
    with pytest.raises(ValueError):
        call()
