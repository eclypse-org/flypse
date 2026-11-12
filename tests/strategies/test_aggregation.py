"""Tests for the aggregation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.federation import Message
from flypse.strategies.aggregation import (
    by_samples,
    hinge_staleness,
    krum,
    mean,
    median,
    normalise,
    polynomial_staleness,
    trimmed_mean,
)


def _vectors(*rows) -> list[np.ndarray]:
    """Return one float vector per row."""
    return [np.asarray(row, dtype=float) for row in rows]


def test_by_samples_reads_the_samples_of_each_contribution():
    """The weight of federated averaging is the number of samples."""
    messages = [
        Message(s, None, {"n_samples": n}, 0.0, 0.0, 0) for s, n in [("a", 3), ("b", 1)]
    ]

    assert by_samples(messages).tolist() == [3.0, 1.0]


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


def test_mean_is_weighted_when_weights_are_given():
    """Without weights the vectors count the same."""
    vectors = _vectors([0.0, 2.0], [2.0, 4.0])

    assert mean(vectors).tolist() == [1.0, 3.0]
    assert mean(vectors, np.array([0.75, 0.25])).tolist() == [0.5, 2.5]


def test_median_and_trimmed_mean_work_by_coordinate():
    """An outlier in each coordinate, from different vectors, is left out."""
    vectors = _vectors([100.0, 1.0], [1.0, 2.0], [2.0, -100.0], [3.0, 3.0], [2.0, 2.0])

    assert median(vectors).tolist() == [2.0, 2.0]
    assert trimmed_mean(vectors, 0.2).tolist() == [7 / 3, 5 / 3]


def test_trimmed_mean_with_nothing_to_drop_is_the_mean():
    """⌊βn⌋ = 0 drops nothing."""
    vectors = _vectors([0.0], [10.0], [2.0])

    assert trimmed_mean(vectors, 0.2).tolist() == mean(vectors).tolist()


@pytest.mark.parametrize("beta", [-0.1, 0.5])
def test_trimmed_mean_refuses_beta_out_of_range(beta):
    """Dropping half of the values at each end would leave nothing."""
    with pytest.raises(ValueError, match="beta"):
        trimmed_mean(_vectors([0.0], [1.0]), beta)


def test_krum_scores_by_squared_distances():
    """With plain distances 8 and 3 would tie; squared distances favour 3."""
    vectors = _vectors([8.0], [9.0], [3.0], [1.0], [5.0])

    assert krum(vectors, f=1).tolist() == [3.0]


def test_multi_krum_leaves_the_byzantine_contributions_out():
    """The mean of the m best vectors excludes a far-away one."""
    rng = np.random.default_rng(0)
    honest = [rng.normal(0.0, 0.1, 10) for _ in range(6)]

    result = krum([*honest, np.full(10, 50.0)], f=1, m=5)

    assert np.abs(result).max() < 1


@pytest.mark.parametrize(("n", "f", "m"), [(4, 1, 1), (5, -1, 1), (5, 1, 0), (5, 1, 6)])
def test_krum_checks_its_preconditions(n, f, m):
    """n > 2f + 2, f >= 0 and 1 <= m <= n."""
    with pytest.raises(ValueError):
        krum(_vectors(*([float(i)] for i in range(n))), f=f, m=m)
