"""Tests for the weighting strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.federation import Message
from flypse.strategies.weighting import (
    FedADP,
    FoolsGold,
    by_samples,
    fedamp,
    hinge_staleness,
    inverse_probability,
    metropolis_hastings,
    min_norm,
    normalise,
    pfedgraph,
    polynomial_staleness,
    uniform_mixing,
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


def test_inverse_probability_inverts_the_selection_probabilities():
    """1 / π_i, for π_i in (0, 1]."""
    assert inverse_probability([0.5, 1.0]).tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="probabilities"):
        inverse_probability([0.0])


GRAPH = {"a": {"b", "c", "d"}, "b": {"a", "c"}, "c": {"a", "b"}, "d": {"a"}}


def _connections(peer: str) -> dict[str, dict[str, int]]:
    """Return the connections of a peer of GRAPH, as get_connections does."""
    return {j: {"degree": len(GRAPH[j])} for j in GRAPH[peer]}


def test_metropolis_hastings_is_doubly_stochastic():
    """Rows sum to 1, self weights are positive, and w_ij = w_ji."""
    weights = {}
    for i in GRAPH:
        own, row = metropolis_hastings(_connections(i))
        assert own > 0
        assert own + sum(row.values()) == pytest.approx(1)
        weights.update({(i, j): w for j, w in row.items()})

    assert all(weights[i, j] == pytest.approx(weights[j, i]) for i, j in weights)


def test_uniform_mixing_weighs_every_model_the_same():
    """1 / (d + 1) for the peer and each neighbour."""
    own, row = uniform_mixing(_connections("a"))

    assert own == 0.25
    assert row == dict.fromkeys(["b", "c", "d"], 0.25)


def test_min_norm_finds_the_common_descent_direction():
    """Two unit vectors at 120 degrees combine with equal weights."""
    a, b = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
    redundant = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]

    assert min_norm([a, b], 1.0) == pytest.approx([0.5, 0.5])
    assert min_norm(redundant, 1.0) == pytest.approx([0.5, 0.5, 0.0], abs=1e-6)


def test_min_norm_stays_within_epsilon_of_the_prior():
    """ε = 0 keeps the prior; a small ε moves each weight by at most ε."""
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    prior = [0.2, 0.2, 0.6]

    assert min_norm(vectors, 0.0, prior) == pytest.approx(prior)
    weights = min_norm(vectors, 0.1, prior, iterations=1)
    assert np.all(np.abs(weights - prior) <= 0.1 + 1e-9)
    assert weights.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    ("epsilon", "prior"), [(-0.1, None), (1.5, None), (0.5, [0.5, 0.6]), (0.5, [1.0])]
)
def test_min_norm_refuses_invalid_values(epsilon, prior):
    """ε in [0, 1], and a prior on the simplex with one weight per vector."""
    with pytest.raises(ValueError):
        min_norm([np.ones(2), np.zeros(2)], epsilon, prior)


def test_fedadp_favours_the_contributions_aligned_with_the_aggregate():
    """A small angle weighs more, and angles are averaged over the rounds."""
    adp = FedADP()
    reference = np.array([1.0, 0.0])

    factors = adp.weigh(
        ["a", "b"], [np.array([1.0, 0.1]), np.array([-1.0, 0.2])], reference
    )
    adp.weigh(["a"], [np.array([0.0, 1.0])], reference)

    assert factors[0] > factors[1] >= 1
    assert adp.rounds == {"a": 2, "b": 1}
    assert adp.angles["a"] == pytest.approx((np.arctan(0.1) + np.pi / 2) / 2)
    with pytest.raises(ValueError, match="alpha"):
        FedADP(alpha=0.0)


def test_foolsgold_weighs_down_sybils_with_the_same_history():
    """Identical histories weigh 0, diverse ones 1."""
    rng = np.random.default_rng(0)
    foolsgold = FoolsGold()
    for _ in range(3):
        sybil = rng.normal(size=10)
        weights = foolsgold.weigh(
            ["s1", "s2", "h1", "h2", "h3"], [sybil, sybil, *rng.normal(size=(3, 10))]
        )

    assert weights.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_foolsgold_gives_no_weight_when_every_history_is_the_same():
    """Nothing tells honest clients from sybils."""
    weights = FoolsGold().weigh(["a", "b"], [np.ones(3), np.ones(3)])

    assert weights.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="kappa"):
        FoolsGold(kappa=0.0)


def test_fedamp_weighs_close_models_more():
    """Rows sum to 1, and a far model gets almost nothing."""
    weights = fedamp(
        [np.zeros(2), np.full(2, 0.1), np.full(2, 3.0)], alpha=0.5, sigma=1.0
    )

    assert weights.sum(axis=1) == pytest.approx(np.ones(3))
    assert weights[0, 1] > 0.4
    assert weights[0, 2] < 1e-6


@pytest.mark.parametrize(("alpha", "sigma"), [(0.0, 1.0), (1.0, 0.0), (5.0, 1.0)])
def test_fedamp_refuses_weights_that_would_be_negative(alpha, sigma):
    """alpha and sigma positive, and no model weighing itself negatively."""
    with pytest.raises(ValueError, match="alpha"):
        fedamp([np.zeros(2), np.zeros(2), np.zeros(2)], alpha, sigma)


def test_pfedgraph_rows_are_on_the_simplex_and_favour_similar_models():
    """Each recipient leans towards the models pointing its way."""
    vectors = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([-1.0, 0.0])]

    weights = pfedgraph(vectors, alpha=1.0)

    assert weights.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(weights >= 0)
    assert weights[0, 1] > weights[0, 2]
    assert pfedgraph(vectors, alpha=100.0, prior=[0.2, 0.3, 0.5])[0] == pytest.approx(
        [0.2, 0.3, 0.5], abs=0.01
    )
    with pytest.raises(ValueError, match="alpha"):
        pfedgraph(vectors, alpha=0.0)
