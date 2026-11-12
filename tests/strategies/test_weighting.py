"""Tests for the weighting strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.federation import Message
from flypse.strategies.weighting import (
    by_samples,
    hinge_staleness,
    normalise,
    polynomial_staleness,
)


def test_by_samples_reads_the_samples_of_each_contribution():
    """The weight of federated averaging is the number of samples."""
    messages = [
        Message(s, None, {"n_samples": n}, 0.0, 0.0, 0) for s, n in [("a", 3), ("b", 1)]
    ]

    assert by_samples(messages).tolist() == [3.0, 1.0]


def test_by_samples_reads_the_samples_under_any_key():
    """Senders may attach the number of samples under another name."""
    messages = [Message("a", None, {"size": 2}, 0.0, 0.0, 0)]

    assert by_samples(messages, sample_size_key="size").tolist() == [2.0]


@pytest.mark.parametrize(
    "staleness", [polynomial_staleness, hinge_staleness], ids=["polynomial", "hinge"]
)
def test_staleness_weights_start_at_one_and_never_grow(staleness):
    """s(0) = 1, non-increasing, and in (0, 1]."""
    s = staleness(range(30))

    assert s[0] == 1
    assert np.all(np.diff(s) <= 0)
    assert np.all((s > 0) & (s <= 1))


def test_hinge_staleness_decays_from_one_after_the_tolerance():
    """Past b, the weight is 1 / (a (τ - b) + 1): the +1 keeps it at most 1."""
    assert hinge_staleness([4, 5], a=10.0, b=4).tolist() == [1.0, 1 / 11]


@pytest.mark.parametrize(
    "call",
    [
        lambda: polynomial_staleness([-1]),
        lambda: polynomial_staleness([0], a=-1.0),
        lambda: hinge_staleness([-1]),
        lambda: hinge_staleness([0], b=-1),
    ],
)
def test_staleness_weights_refuse_negative_values(call):
    """A negative staleness, or parameter, would give a weight above 1."""
    with pytest.raises(ValueError, match="non-negative"):
        call()


def test_normalise_multiplies_the_factors_and_sums_to_one():
    """Weights are the product of the factors, normalised over the contributions."""
    weights = normalise(np.array([3.0, 1.0]), np.array([1.0, 0.5]))

    assert np.allclose(weights, [6 / 7, 1 / 7])


@pytest.mark.parametrize(
    "factors",
    [(), (np.array([1.0, -1.0]),), (np.zeros(2),)],
    ids=["none", "negative", "zero"],
)
def test_normalise_refuses_weights_that_cannot_be_normalised(factors):
    """No factors, a negative weight, or all zero weights are refused."""
    with pytest.raises(ValueError):
        normalise(*factors)
