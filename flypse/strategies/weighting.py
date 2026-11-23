"""Module for the weighting strategies.

Weighting decides how much each contribution counts. A weight is a product of
factors, normalised to sum to 1 by :func:`normalise`: e.g., the number of
samples n_i (federated averaging), the staleness (FedBuff), the trust
(FLTrust), the fairness (q-FFL). In decentralised learning, the mixing row of a
peer (e.g., Metropolis-Hastings) is the same decision. It is the relevance role
of p_i, whose participation role is :mod:`~flypse.strategies.selection`.

A factor reads the contributions, received
:class:`~flypse.federation.message.Message` objects, or facts about them, such
as their staleness from :meth:`~flypse.federation.model.FederatedModel.lag`.

Some weights depend on the recipient: a personalised model per contribution
(FedAMP, pFedGraph) comes from a matrix whose row i weighs the contributions
for recipient i, ``aggregation.mean(vectors, weights[i])``. A mixing row weighs
a peer's own model and its neighbours', from ``await self.get_connections()``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from flypse.strategies.screening import (
    _cosine_similarities,
    _squared_distances,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

    from flypse.federation.message import Message


def by_samples(
    messages: Sequence[Message], sample_size_key: str = "n_samples"
) -> np.ndarray:
    """Weigh each contribution by the number of samples it was trained on.

    It is the weighting of federated averaging, Θ = Σ_c (n_c / n) Θ_c.

    Args:
        messages (Sequence[Message]): The contributions, each with its number
            of samples in its data.
        sample_size_key (str): The key of the number of samples in the data
            of a contribution. Defaults to ``"n_samples"``.

    Returns:
        np.ndarray: The number of samples of each contribution.
    """
    return np.array(
        [message.data[sample_size_key] for message in messages], dtype=float
    )


def polynomial_staleness(lags: Sequence[int], a: float = 0.5) -> np.ndarray:
    """Weigh each contribution by its staleness, as s(τ) = (1 + τ)^-a.

    Args:
        lags (Sequence[int]): The staleness τ of each contribution, e.g. from
            :meth:`~flypse.federation.model.FederatedModel.lag`.
        a (float): How fast the weight decays. Defaults to 0.5.

    Returns:
        np.ndarray: The factors, 1 for a fresh contribution and decreasing in
        (0, 1] with the staleness.

    Raises:
        ValueError: If a staleness is negative, or ``a`` is.
    """
    if a < 0:
        raise ValueError(f"a must be non-negative, got {a}")
    return (1.0 + _staleness(lags)) ** -a


def hinge_staleness(lags: Sequence[int], a: float = 10.0, b: int = 4) -> np.ndarray:
    """Weigh each contribution by its staleness, as 1 up to ``b``, then decaying.

    s(τ) = 1 if τ ≤ b, else 1 / (a (τ - b) + 1).

    Args:
        lags (Sequence[int]): The staleness τ of each contribution.
        a (float): How fast the weight decays after ``b``. Defaults to 10.0.
        b (int): The staleness tolerated at full weight. Defaults to 4.

    Returns:
        np.ndarray: The factors, in (0, 1] and non-increasing with the
        staleness.

    Raises:
        ValueError: If a staleness is negative, or ``a`` or ``b`` is.
    """
    if a < 0 or b < 0:
        raise ValueError(f"a and b must be non-negative, got {a} and {b}")
    tau = _staleness(lags)
    return np.where(tau <= b, 1.0, 1.0 / (a * (tau - b) + 1.0))


def normalise(*factors: np.ndarray) -> np.ndarray:
    """Multiply the weight factors of the contributions and normalise them.

    Args:
        *factors (np.ndarray): One factor per contribution each, e.g.
            :func:`by_samples` and a staleness.

    Returns:
        np.ndarray: The weights, non-negative and summing to 1.

    Raises:
        ValueError: If no factor is given, a weight is negative, or all are 0.

    Example:
        .. code-block:: python

            lags = [self.model.lag(m.record, "global") for m in messages]
            weights = normalise(by_samples(messages), polynomial_staleness(lags))
    """
    if not factors:
        raise ValueError("no weight factors to normalise")
    weights = np.prod(np.vstack(factors), axis=0)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    return weights / weights.sum()


def inverse_probability(probabilities: Sequence[float]) -> np.ndarray:
    """Weigh each contribution by the inverse of its probability to be selected.

    Multiplied by the relevance p_i (e.g., the share of samples of its sender
    over all the candidates), it is p_i / π_i. Normalised by
    :func:`normalise`, the mean is the ratio estimator: consistent, with a lower
    variance; not normalised, ``np.sum(weights[:, None] * vectors, axis=0)`` is
    the unbiased Horvitz-Thompson estimator of the mean over all candidates.

    Args:
        probabilities (Sequence[float]): The probability π_i of each
            contribution's sender to be selected, as returned by
            :mod:`~flypse.strategies.selection`.

    Returns:
        np.ndarray: The factors 1 / π_i.

    Raises:
        ValueError: If a probability is not in (0, 1].
    """
    pi = np.asarray(probabilities, dtype=float)
    if np.any((pi <= 0) | (pi > 1)):
        raise ValueError("probabilities must be in (0, 1]")
    return 1 / pi


def metropolis_hastings(
    connections: Mapping[str, Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Return the Metropolis-Hastings mixing row of a peer.

    w_ij = 1 / (1 + max(d_i, d_j)) for each neighbour j, w_ii = 1 - Σ_j w_ij:
    doubly stochastic on an undirected graph, so the peers average to the
    average of their models.

    Args:
        connections (Mapping[str, Mapping[str, Any]]): The connections of the
            peer, ``await self.get_connections()``, with the ``degree`` of
            each neighbour.

    Returns:
        tuple[float, dict[str, float]]: The weight of the peer's own model,
        and the weight of each neighbour's.
    """
    degree = len(connections)
    row = {j: 1 / (1 + max(degree, c["degree"])) for j, c in connections.items()}
    return 1 - sum(row.values()), row


def uniform_mixing(
    connections: Mapping[str, Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Return the uniform mixing row of a peer: 1 / (d_i + 1) for each model.

    Unless every peer has the same degree, the rows are stochastic but their
    columns are not: the peers then reach a consensus weighted by their
    degrees, not the plain average (see :func:`metropolis_hastings`).

    Args:
        connections (Mapping[str, Mapping[str, Any]]): The connections of the
            peer, ``await self.get_connections()``.

    Returns:
        tuple[float, dict[str, float]]: The weight of the peer's own model,
        and the weight of each neighbour's.
    """
    weight = 1 / (len(connections) + 1)
    return weight, dict.fromkeys(connections, weight)


def min_norm(
    vectors: Sequence[np.ndarray],
    epsilon: float,
    prior: Sequence[float] | None = None,
    iterations: int = 1000,
) -> np.ndarray:
    """Weigh the contributions to the combination of minimum norm (FedMGDA+).

    The weights λ minimise ‖Σ_i λ_i v_i‖² on the simplex, within ε of the prior
    for each contribution (Hu et al., 2022): the common descent direction of
    the contributions, the one that hurts none of them. FedMGDA+ normalises
    the updates to unit norm first, and averages those with the weights.
    ε = 1 is plain MGDA; ε = 0 keeps the prior.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions,
            e.g. their updates normalised to unit norm.
        epsilon (float): How far each weight may move from the prior, in
            [0, 1].
        prior (Sequence[float] | None): The prior weights, on the simplex.
            Defaults to None, meaning equal weights.
        iterations (int): The largest number of steps of projected gradient
            descent. Defaults to 1000.

    Returns:
        np.ndarray: The weights, non-negative and summing to 1.

    Raises:
        ValueError: If ``epsilon`` is not in [0, 1], or the prior is not on
            the simplex.
    """
    stacked = np.vstack(vectors).astype(float)
    n = len(stacked)
    p = np.full(n, 1 / n) if prior is None else np.asarray(prior, dtype=float)
    if not 0 <= epsilon <= 1:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if p.shape != (n,) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("the prior must be one weight per vector, summing to 1")
    low, high = np.maximum(p - epsilon, 0), np.minimum(p + epsilon, 1)
    gram = stacked @ stacked.T
    step = 1 / (2 * max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12))
    weights = p
    for _ in range(iterations):
        moved = _project(weights - step * 2 * gram @ weights, low, high)
        if np.allclose(moved, weights, rtol=0, atol=1e-12):
            break
        weights = moved
    return weights


class FedADP:
    """Weigh each contribution by its angle with the aggregate (FedADP).

    The angle θ_i between a contribution and the aggregate is smoothed over the
    rounds its sender took part in, θ̃_i ← ((t - 1) θ̃_i + θ_i) / t, and mapped
    by the Gompertz function f(θ) = alpha (1 - exp(-exp(-alpha (θ - 1)))): the factor
    exp(f(θ̃_i)) favours the contributions that agree with the others (Wu and
    Wang, 2021). FedADP multiplies it by :func:`by_samples`.
    """

    def __init__(self, alpha: float = 5.0):
        """Initialise the weighting.

        Args:
            alpha (float): The steepness of the Gompertz function. Defaults
                to 5.0.

        Raises:
            ValueError: If ``alpha`` is not positive.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.angles: dict[str, float] = {}
        """The smoothed angle of each sender, in radians."""
        self.rounds: dict[str, int] = {}
        """The number of rounds each sender took part in."""

    def weigh(
        self,
        senders: Sequence[str],
        vectors: Sequence[np.ndarray],
        reference: np.ndarray,
    ) -> np.ndarray:
        """Weigh the contributions of a round, updating the smoothed angles.

        Args:
            senders (Sequence[str]): The sender of each contribution.
            vectors (Sequence[np.ndarray]): The updates of the contributions.
            reference (np.ndarray): The aggregate update, e.g. the mean of
                the updates weighted by their samples.

        Returns:
            np.ndarray: The factors exp(f(θ̃_i)).
        """
        factors = []
        for sender, vector in zip(senders, vectors, strict=True):
            cosine = _cosine_similarities([vector, reference])[0, 1]
            t = self.rounds.get(sender, 0) + 1
            angle = float(np.arccos(np.clip(cosine, -1, 1)))
            self.angles[sender] = ((t - 1) * self.angles.get(sender, 0.0) + angle) / t
            self.rounds[sender] = t
            gompertz = 1 - np.exp(-np.exp(-self.alpha * (self.angles[sender] - 1)))
            factors.append(np.exp(self.alpha * gompertz))
        return np.array(factors)


class FoolsGold:
    """Weigh down the contributions whose history is similar to another's.

    Sybils pushing the same poisoned update look alike over the rounds, while
    honest clients diverge (Fung et al., 2020). The weight of a sender falls
    with the largest cosine similarity between its summed updates and another
    sender's, pardoning the similarity to a sender that is itself less
    similar to the others; then it is rescaled by a logit, κ (ln(w / (1 - w)) +
    0.5), clipped to [0, 1]. Under IID data honest clients look alike too, and
    lose weight.
    """

    def __init__(self, kappa: float = 1.0):
        """Initialise the weighting.

        Args:
            kappa (float): The confidence κ of the logit. Defaults to 1.0.

        Raises:
            ValueError: If ``kappa`` is not positive.
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        self.kappa = kappa
        self.history: dict[str, np.ndarray] = {}
        """The sum of the updates of each sender."""

    def weigh(
        self, senders: Sequence[str], vectors: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Weigh the contributions of a round, adding them to the history.

        Args:
            senders (Sequence[str]): The sender of each contribution, each
                once.
            vectors (Sequence[np.ndarray]): The updates of the contributions.

        Returns:
            np.ndarray: The factors, in [0, 1]; all 0 if every history is the
            same.
        """
        for sender, vector in zip(senders, vectors, strict=True):
            self.history[sender] = self.history.get(sender, 0) + np.asarray(vector)
        similarity = _cosine_similarities([self.history[s] for s in senders])
        similarity -= np.eye(len(senders))
        top = similarity.max(axis=1)
        # pardon i for resembling j when j resembles someone even more
        pardoned = (top[None, :] > top[:, None]) & (top[None, :] > 0)
        ratio = np.divide(
            top[:, None], top[None, :], where=pardoned, out=np.ones_like(similarity)
        )
        weights = np.clip(1 - (similarity * ratio).max(axis=1), 0, 1)
        if weights.max() == 0:
            return weights
        weights = np.minimum(weights / weights.max(), 0.99)
        with np.errstate(divide="ignore"):
            logit = self.kappa * (np.log(weights / (1 - weights)) + 0.5)
        return np.clip(logit, 0, 1)


def fedamp(vectors: Sequence[np.ndarray], alpha: float, sigma: float) -> np.ndarray:
    """Weigh the models for each recipient by their closeness to its own (FedAMP).

    w_ij = alpha exp(-‖v_i - v_j‖² / sigma) / sigma for j ≠ i, and w_ii = 1 - Σ_j w_ij:
    each client gets a personalised model mostly made of the models close to
    its own (Huang et al., 2021).

    Args:
        vectors (Sequence[np.ndarray]): The models of the clients.
        alpha (float): The step, small enough that every w_ii stays
            non-negative.
        sigma (float): The scale of the distances.

    Returns:
        np.ndarray: The weights, row i for recipient i, each row summing to 1.

    Raises:
        ValueError: If ``alpha`` or ``sigma`` is not positive, or ``alpha``
            makes a weight negative.
    """
    if alpha <= 0 or sigma <= 0:
        raise ValueError(f"alpha and sigma must be positive, got {alpha}, {sigma}")
    weights = alpha * np.exp(-_squared_distances(vectors) / sigma) / sigma
    np.fill_diagonal(weights, 0)
    np.fill_diagonal(weights, 1 - weights.sum(axis=1))
    if np.any(np.diag(weights) < 0):
        raise ValueError("alpha is too large: a model would weigh itself negatively")
    return weights


def heurfedamp(
    vectors: Sequence[np.ndarray], sigma: float, self_weight: float
) -> np.ndarray:
    """Weigh the models for each recipient by a softmax of their angles (HeurFedAMP).

    w_ii = self_weight, and the rest, 1 - self_weight, is split among the
    others in proportion to exp(sigma cos(v_i, v_j)): the heuristic variant of
    :func:`fedamp` (Huang et al., 2021), comparing angles instead of distances,
    with no weight that can turn negative. A single model weighs only itself.

    Args:
        vectors (Sequence[np.ndarray]): The models of the clients, or their
            updates (e.g., sigma = 10 and self_weight = 0.3 on updates in the
            comparisons of pFedGraph).
        sigma (float): The sharpness of the softmax: 0 splits the rest
            equally.
        self_weight (float): The weight of each recipient's own model, in
            [0, 1].

    Returns:
        np.ndarray: The weights, row i for recipient i, each row summing to 1.

    Raises:
        ValueError: If ``sigma`` is negative or ``self_weight`` is not in
            [0, 1].
    """
    if sigma < 0 or not 0 <= self_weight <= 1:
        raise ValueError(
            "sigma must be non-negative and self_weight in [0, 1], "
            f"got {sigma}, {self_weight}"
        )
    if len(vectors) == 1:
        return np.ones((1, 1))
    logits = sigma * _cosine_similarities(vectors)
    np.fill_diagonal(logits, -np.inf)
    # shifted by the largest of each row, so that exp cannot overflow
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights *= (1 - self_weight) / weights.sum(axis=1, keepdims=True)
    np.fill_diagonal(weights, self_weight)
    return weights


def pfedgraph(
    vectors: Sequence[np.ndarray],
    alpha: float = 0.8,
    prior: Sequence[float] | None = None,
) -> np.ndarray:
    """Weigh the models for each recipient by a learnt collaboration graph (pFedGraph).

    Row i minimises alpha ‖w_i - p‖² - Σ_j w_ij cos(v_i, v_j) on the simplex (Ye et
    al., 2023): close to the prior p (e.g., the shares of samples), with more
    weight on the models similar to v_i. Its solution is the projection of
    p + cos(v_i, ·) / (2 alpha) on the simplex. The reference code also rounds
    every cosine above 0.9 up to 1, a step the objective does not have: it is
    left out.

    Args:
        vectors (Sequence[np.ndarray]): The updates of the clients: each
            model minus the one it started the round from. The models
            themselves share their start, so their cosines are all near 1.
        alpha (float): The pull towards the prior. Defaults to 0.8, as in the
            reference code.
        prior (Sequence[float] | None): The prior weights, on the simplex.
            Defaults to None, meaning equal weights.

    Returns:
        np.ndarray: The weights, row i for recipient i, each row on the
        simplex.

    Raises:
        ValueError: If ``alpha`` is not positive.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    similarity = _cosine_similarities(vectors)
    n = len(similarity)
    p = np.full(n, 1 / n) if prior is None else np.asarray(prior, dtype=float)
    zeros, ones = np.zeros(n), np.ones(n)
    return np.vstack(
        [_project(p + row / (2 * alpha), zeros, ones) for row in similarity]
    )


def _project(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Project a vector on the simplex, within bounds for each coordinate.

    The projection is clip(y - τ, low, high) for the τ making it sum to 1,
    found by bisection.

    Args:
        y (np.ndarray): The vector.
        low (np.ndarray): The lower bounds, summing to at most 1.
        high (np.ndarray): The upper bounds, summing to at least 1.

    Returns:
        np.ndarray: The projection.
    """
    below, above = float(np.min(y - high)), float(np.max(y - low))
    for _ in range(100):
        tau = (below + above) / 2
        if np.clip(y - tau, low, high).sum() > 1:
            below = tau
        else:
            above = tau
    return np.clip(y - (below + above) / 2, low, high)


def _staleness(lags: Sequence[int]) -> np.ndarray:
    """Validate the staleness of the contributions.

    Args:
        lags (Sequence[int]): The staleness of each contribution.

    Returns:
        np.ndarray: The staleness, as floats.

    Raises:
        ValueError: If a staleness is negative.
    """
    tau = np.asarray(lags, dtype=float)
    if np.any(tau < 0):
        raise ValueError("staleness must be non-negative")
    return tau
