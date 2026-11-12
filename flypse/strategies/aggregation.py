"""Module for the aggregation strategies.

Aggregating is weighting the contributions an entity received, then reducing
their vectors to one. A contribution is a received
:class:`~flypse.federation.message.Message`; its vector is the model it carries
(e.g., ``message.record.parameters``) or its difference from the model it was
generated from (see :meth:`~flypse.federation.model.FederatedModel.base`).

Weights are factors over the contributions, multiplied and normalised by
:func:`normalise`. The robust reductions (:func:`median`, :func:`trimmed_mean`,
:func:`krum`) take no weights: weighting would let one contribution outweigh the
others, which is what they guard against.

Float sums depend on their order: sort the contributions (e.g., by sender) for
a result that does not follow their arrival.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flypse.federation.message import Message

MAX_TRIM = 0.5
"""The bound, excluded, of the fraction ``trimmed_mean`` drops at each end."""


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


def mean(
    vectors: Sequence[np.ndarray], weights: np.ndarray | None = None
) -> np.ndarray:
    """Average the vectors.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        weights (np.ndarray | None): The weight of each vector, e.g. from
            :func:`normalise`. Defaults to None, meaning equal weights.

    Returns:
        np.ndarray: The (weighted) mean.
    """
    return np.average(np.vstack(vectors), axis=0, weights=weights)


def median(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Take the median of each coordinate of the vectors.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.

    Returns:
        np.ndarray: The coordinate-wise median.
    """
    return np.median(np.vstack(vectors), axis=0)


def trimmed_mean(vectors: Sequence[np.ndarray], beta: float) -> np.ndarray:
    """Average each coordinate without its most extreme values.

    For each coordinate, the ⌊βn⌋ smallest and the ⌊βn⌋ largest of the n values
    are dropped. With fewer than 1/β vectors nothing is dropped, and it is the
    mean.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        beta (float): The fraction dropped at each end, in [0, 0.5).

    Returns:
        np.ndarray: The coordinate-wise trimmed mean.

    Raises:
        ValueError: If ``beta`` is not in [0, 0.5).
    """
    if not 0 <= beta < MAX_TRIM:
        raise ValueError(f"beta must be in [0, 0.5), got {beta}")
    stacked = np.sort(np.vstack(vectors), axis=0)
    k = int(beta * len(stacked))
    return stacked[k : len(stacked) - k].mean(axis=0)


def krum(vectors: Sequence[np.ndarray], f: int, m: int = 1) -> np.ndarray:
    """Average the m vectors closest to their neighbours (Multi-Krum).

    The score of a vector is the sum of its squared distances to its n - f - 2
    nearest other vectors; the m vectors with the lowest scores are averaged.
    With m = 1 it is Krum.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.
        m (int): The number of vectors averaged. Defaults to 1.

    Returns:
        np.ndarray: The mean of the selected vectors.

    Raises:
        ValueError: If there are not more than 2f + 2 vectors, ``f`` is
            negative, or ``m`` is not between 1 and the number of vectors.
    """
    n = len(vectors)
    if f < 0 or n <= 2 * f + 2:
        raise ValueError(f"Krum requires f >= 0 and n > 2f + 2, got n={n}, f={f}")
    if not 1 <= m <= n:
        raise ValueError(f"m must be between 1 and {n}, got {m}")
    stacked = np.vstack(vectors)
    squared = ((stacked[:, None, :] - stacked[None, :, :]) ** 2).sum(axis=-1)
    # the first column of each sorted row is the distance of a vector to itself
    scores = np.sort(squared, axis=1)[:, 1 : n - f - 1].sum(axis=1)
    return stacked[np.argsort(scores, kind="stable")[:m]].mean(axis=0)


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
