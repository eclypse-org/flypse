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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

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
