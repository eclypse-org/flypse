"""Module for the selection strategies.

Selection decides whom to engage: which entities, among the candidates, are
contacted at all. E.g., federated averaging samples 10 of 100 clients at
random, a gossip peer picks one of its neighbours, Oort prefers clients that
are both useful and fast. Not being selected means not being contacted: no
message, and no state on the entity.

A strategy returns the selected entities and the probability of each candidate
to be selected: an unbiased aggregate weighs a contribution by p_i / π_i, and
privacy accounting needs the actual scheme. It is the participation role of
p_i, whose relevance role is :mod:`~flypse.strategies.weighting`.

The candidates usually are the neighbours of the entity
(``await self.get_neighbors()``), and the randomness its own ``rng``, seeded by
the process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import random
    from collections.abc import (
        Collection,
        Mapping,
    )


def uniform(
    candidates: Collection[str], k: int, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Select k of the candidates, uniformly and without replacement.

    Args:
        candidates (Collection[str]): The identifiers of the candidates.
        k (int): How many to select; all of them if there are fewer.
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The selected candidates, and the
        probability of each candidate to be selected, k / n.

    Raises:
        ValueError: If ``k`` is negative.

    Example:
        .. code-block:: python

            selected, _ = uniform(await self.get_neighbors(), 10, self.rng)
            await self.send(selected, self.model.versions[-1])
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    # sorted, so that the selection depends on the seed and not on the order given
    pool = sorted(candidates)
    k = min(k, len(pool))
    return rng.sample(pool, k), {c: k / len(pool) for c in pool}


def poisson(
    candidates: Collection[str], q: float, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Select each candidate independently, with probability q.

    The number of selected candidates varies, and can be 0.

    Args:
        candidates (Collection[str]): The identifiers of the candidates.
        q (float): The probability of each candidate to be selected, in
            [0, 1].
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The selected candidates, and the
        probability of each candidate to be selected, q.

    Raises:
        ValueError: If ``q`` is not in [0, 1].
    """
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    pool = sorted(candidates)
    return [c for c in pool if rng.random() < q], dict.fromkeys(pool, q)


def proportional(
    sizes: Mapping[str, float], k: int, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Draw k candidates with replacement, each with probability n_i / n.

    It is the sampling of the convergence analysis of federated averaging on
    non-IID data (Li et al., 2020): a candidate drawn twice appears twice, and
    the contributions are averaged with equal weights, one per draw.

    Args:
        sizes (Mapping[str, float]): The size n_i of each candidate, e.g. its
            number of samples as last reported.
        k (int): The number of draws.
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The drawn candidates, and the
        probability of each candidate to be drawn at least once,
        1 - (1 - n_i / n)^k.

    Raises:
        ValueError: If ``k`` is negative, a size is negative, or all are 0.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    pool, p = _shares(sizes)
    drawn = rng.choices(pool, weights=p.tolist(), k=k)
    return drawn, {
        c: float(1 - (1 - share) ** k) for c, share in zip(pool, p, strict=True)
    }


def round_robin(
    candidates: Collection[str], k: int, turn: int
) -> tuple[list[str], dict[str, float]]:
    """Select the candidates k at a time, in turn.

    Turn t selects the k candidates after the ones of turn t - 1, wrapping
    around: every candidate is selected once every ⌈n / k⌉ turns (cyclic
    participation).

    Args:
        candidates (Collection[str]): The identifiers of the candidates.
        k (int): How many to select; all of them if there are fewer.
        turn (int): The number of the turn, e.g. the global version of the
            model.

    Returns:
        tuple[list[str], dict[str, float]]: The selected candidates, and the
        probability of each candidate to be selected: 1 or 0, as the
        selection is not random.

    Raises:
        ValueError: If ``k`` or ``turn`` is negative.
    """
    if k < 0 or turn < 0:
        raise ValueError(f"k and turn must be non-negative, got {k}, {turn}")
    pool = sorted(candidates)
    k = min(k, len(pool))
    start = turn * k % len(pool) if pool else 0
    selected = [pool[(start + j) % len(pool)] for j in range(k)]
    return selected, {c: float(c in selected) for c in pool}


def clustered(
    sizes: Mapping[str, float], m: int, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Draw one candidate from each of m clusters of equal size (Fraboni et al., 2021).

    The candidates, largest first, fill m clusters, each with a share 1 / m of
    the total size n: a candidate of size n_i enters the clusters with a total
    weight m n_i / n, possibly split between consecutive clusters. One
    candidate is drawn from each cluster, in proportion to its weight there.
    Every candidate has the same expected number of draws as with
    :func:`proportional`, with a lower variance: the contributions are
    averaged the same way, with equal weights, one per draw.

    Args:
        sizes (Mapping[str, float]): The size n_i of each candidate, e.g. its
            number of samples as last reported.
        m (int): The number of clusters, hence of draws.
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The drawn candidates, one per
        cluster, and the probability of each candidate to be drawn at least
        once.

    Raises:
        ValueError: If ``m`` is negative, a size is negative, or all are 0.
    """
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    pool, p = _shares(sizes, largest_first=True)
    end = np.cumsum(m * p)
    start = end - m * p
    edges = np.arange(m)[:, None]
    # the weight of candidate i in cluster c: the overlap of its share with [c, c + 1)
    weights = np.clip(np.minimum(end, edges + 1) - np.maximum(start, edges), 0, 1)
    drawn = [rng.choices(pool, weights=row.tolist())[0] for row in weights]
    missed = np.prod(1 - weights, axis=0)
    return drawn, {c: float(1 - miss) for c, miss in zip(pool, missed, strict=True)}


def _shares(
    sizes: Mapping[str, float], largest_first: bool = False
) -> tuple[list[str], np.ndarray]:
    """Return the candidates and their shares n_i / n of the total size.

    Args:
        sizes (Mapping[str, float]): The size of each candidate.
        largest_first (bool): Whether to sort the candidates by decreasing
            size rather than by identifier. Defaults to False.

    Returns:
        tuple[list[str], np.ndarray]: The candidates, sorted, and their shares.

    Raises:
        ValueError: If a size is negative, or all are 0.
    """
    pool = sorted(sizes)
    if largest_first:
        pool.sort(key=lambda c: -sizes[c])  # stable: ties stay by identifier
    n = np.array([sizes[c] for c in pool], dtype=float)
    if np.any(n < 0) or n.sum() <= 0:
        raise ValueError("sizes must be non-negative and not all zero")
    return pool, n / n.sum()
