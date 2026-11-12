"""Module for the aggregation strategies.

Aggregation combines the contributions an entity uses into one vector, with an
aggregation rule: e.g., the weighted mean, the coordinate-wise median, the
trimmed mean. The vector of a contribution is the model it carries (e.g.,
``message.record.parameters``) or its difference from the model it was
generated from (see :meth:`~flypse.federation.model.FederatedModel.base`).

It is apart from :mod:`~flypse.strategies.weighting` because the same weights
can feed different rules, and most robust rules (:func:`median`,
:func:`trimmed_mean`, :func:`meamed`) ignore weights on purpose: weighting
would let one contribution outweigh the others, which is what they guard
against.

Float sums depend on their order: sort the contributions (e.g., by sender) for
a result that does not follow their arrival.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_TRIM = 0.5
"""The bound, excluded, of the fraction ``trimmed_mean`` drops at each end."""


def mean(
    vectors: Sequence[np.ndarray], weights: np.ndarray | None = None
) -> np.ndarray:
    """Average the vectors.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        weights (np.ndarray | None): The weight of each vector, e.g. from
            :func:`~flypse.strategies.weighting.normalise`. Defaults to None,
            meaning equal weights.

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


def meamed(vectors: Sequence[np.ndarray], f: int) -> np.ndarray:
    """Average, for each coordinate, the n - f values closest to the median.

    It is the mean around the median of Xie et al. (2018), and the last step of
    Bulyan, with 2f for f (see :func:`~flypse.strategies.screening.bulyan`).

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of values dropped for each coordinate.

    Returns:
        np.ndarray: The coordinate-wise mean around the median.

    Raises:
        ValueError: If ``f`` is not between 0 and n - 1.
    """
    stacked = np.vstack(vectors)
    n = len(stacked)
    if not 0 <= f < n:
        raise ValueError(f"f must be between 0 and {n - 1}, got {f}")
    distance = np.abs(stacked - np.median(stacked, axis=0))
    closest = np.argsort(distance, axis=0, kind="stable")[: n - f]
    return np.take_along_axis(stacked, closest, axis=0).mean(axis=0)


def geometric_median(
    vectors: Sequence[np.ndarray],
    weights: np.ndarray | None = None,
    iterations: int = 100,
    nu: float = 1e-6,
) -> np.ndarray:
    """Approximate the vector minimising the weighted sum of distances (RFA).

    The smoothed Weiszfeld algorithm of robust federated aggregation (Pillutla
    et al., 2022) starts from the weighted mean and repeats
    z ← Σ β_i v_i / Σ β_i, with β_i = w_i / max(nu, ‖z - v_i‖), until z moves by
    less than nu. Unlike the other robust rules it honours the weights.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        weights (np.ndarray | None): The weight of each vector. Defaults to
            None, meaning equal weights.
        iterations (int): The largest number of iterations. Defaults to 100.
        nu (float): The smoothing, which also stops the iterations.
            Defaults to 1e-6.

    Returns:
        np.ndarray: The approximate geometric median.

    Raises:
        ValueError: If ``nu`` is not positive.
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    stacked = np.vstack(vectors).astype(float)
    w = np.ones(len(stacked)) if weights is None else np.asarray(weights, dtype=float)
    median = np.average(stacked, axis=0, weights=w)
    for _ in range(iterations):
        beta = w / np.maximum(nu, np.linalg.norm(stacked - median, axis=1))
        moved = beta @ stacked / beta.sum()
        if np.linalg.norm(moved - median) < nu:
            return moved
        median = moved
    return median


def centered_clipping(
    vectors: Sequence[np.ndarray],
    reference: np.ndarray,
    tau: float,
    iterations: int = 1,
) -> np.ndarray:
    """Move from a reference by the mean of the clipped differences (CC).

    z ← z + (1/n) Σ_i (v_i - z) min(1, τ / ‖v_i - z‖), starting from the
    reference: no contribution moves the result by more than τ / n
    (Karimireddy et al., 2021). The reference is the previous aggregate, kept
    by the entity: clipping around anything else (e.g., a random vector) loses
    the guarantee.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        reference (np.ndarray): The previous aggregate.
        tau (float): The clipping radius τ.
        iterations (int): The number of clipping steps. Defaults to 1.

    Returns:
        np.ndarray: The clipped aggregate.

    Raises:
        ValueError: If ``tau`` is negative.
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    stacked = np.vstack(vectors).astype(float)
    result = np.asarray(reference, dtype=float)
    for _ in range(iterations):
        difference = stacked - result
        norms = np.linalg.norm(difference, axis=1)
        scale = np.minimum(
            1, np.divide(tau, norms, where=norms > 0, out=np.ones_like(norms))
        )
        result = result + (scale[:, None] * difference).mean(axis=0)
    return result


def sign_majority(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Take the majority vote of the signs of each coordinate (signSGD).

    Each contribution votes with the sign of its coordinates; the result is the
    sign of the votes, 0 on a tie (Bernstein et al., 2019). The entity steps
    along it with its own learning rate. The signs are biased unless the
    learners compress them with error feedback.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions, or
            their signs.

    Returns:
        np.ndarray: The vector of the majority signs, in {-1, 0, 1}.
    """
    return np.sign(np.sign(np.vstack(vectors)).sum(axis=0))


def masked_mean(
    vectors: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    previous: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average each coordinate over the contributions that cover it.

    Each contribution covers part of the coordinates (e.g., the sub-model a
    weak device trained, as in HeteroFL); a coordinate is averaged over the
    contributions covering it, and one that none covers keeps its previous
    value, never 0.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        masks (Sequence[np.ndarray]): The coordinates each contribution
            covers, as boolean vectors.
        previous (np.ndarray): The value of each coordinate before, e.g. the
            entity's own model.
        weights (np.ndarray | None): The weight of each vector. Defaults to
            None, meaning equal weights.

    Returns:
        np.ndarray: The masked mean.
    """
    stacked = np.vstack(vectors).astype(float)
    covered = np.vstack(masks).astype(float)
    w = np.ones(len(stacked)) if weights is None else np.asarray(weights, dtype=float)
    total = w @ covered
    summed = w @ (covered * stacked)
    return np.where(total > 0, summed / np.where(total > 0, total, 1), previous)
