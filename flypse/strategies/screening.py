"""Module for the screening strategies.

Screening decides which of the received contributions to use, and in which
groups. A filter keeps some and drops the others: e.g., a norm filter drops
suspiciously large updates, Krum keeps the contributions closest to the rest.
Clustering splits them into groups, each combined on its own and sent back to
its members, as in clustered federated learning (CFL). Either way, it is a
partition of the contributions.

A filter returns the positions of the contributions it keeps, in increasing
order, so that their vectors, senders and weights stay aligned:

```python
kept = krum(deltas, f=1, m=len(deltas) - 1)
update = aggregation.mean([deltas[i] for i in kept])
```

A grouping returns groups of positions, each combined on its own: the clusters of
[cosine_bipartition][flypse.strategies.screening.cosine_bipartition], the buckets of
[bucketing][flypse.strategies.screening.bucketing], or the overlapping neighbourhoods of
[nnm][flypse.strategies.screening.nnm].
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def krum(vectors: Sequence[np.ndarray], f: int, m: int = 1) -> list[int]:
    """Keep the m vectors closest to their neighbours.

    The score of a vector is the sum of its squared distances to its ``n - f - 2``
    nearest other vectors; the m vectors with the lowest scores are kept. Krum
    keeps one; Multi-Krum keeps m and averages them (Blanchard et al., 2017).

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.
        m (int): The number of vectors kept. Defaults to 1.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.

    Raises:
        ValueError: If there are not more than 2f + 2 vectors, ``f`` is
            negative, or ``m`` is not between 1 and the number of vectors.
    """
    n = len(vectors)
    if f < 0 or n <= 2 * f + 2:
        raise ValueError(f"Krum requires f >= 0 and n > 2f + 2, got n={n}, f={f}")
    if not 1 <= m <= n:
        raise ValueError(f"m must be between 1 and {n}, got {m}")
    scores = _krum_scores(_squared_distances(vectors), f)
    return sorted(np.argsort(scores, kind="stable")[:m].tolist())


def bulyan(vectors: Sequence[np.ndarray], f: int) -> list[int]:
    """Keep ``θ = n - 2f`` vectors, chosen one at a time by Krum.

    Each choice scores the vectors not chosen yet, among themselves, and keeps
    the best (El Mhamdi et al., 2018). Bulyan then averages, coordinate by
    coordinate, the ``θ - 2f`` values closest to the median:
    ``aggregation.meamed([vectors[i] for i in kept], 2 * f)``.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.

    Raises:
        ValueError: If there are fewer than 4f + 3 vectors, or ``f`` is
            negative.
    """
    n = len(vectors)
    if f < 0 or n < 4 * f + 3:
        raise ValueError(f"Bulyan requires f >= 0 and n >= 4f + 3, got n={n}, f={f}")
    squared = _squared_distances(vectors)
    left = list(range(n))
    kept = []
    for _ in range(n - 2 * f):
        scores = _krum_scores(squared[np.ix_(left, left)], f)
        kept.append(left.pop(int(np.argmin(scores))))
    return sorted(kept)


def mda(vectors: Sequence[np.ndarray], f: int) -> list[int]:
    """Keep the n - f vectors of smallest diameter (minimum diameter averaging).

    The diameter of a set of vectors is the largest distance between two of
    them. Every subset of n - f vectors is tried, so the cost grows as the
    binomial coefficient of n and f: it suits a few tens of contributions.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.

    Raises:
        ValueError: If ``f`` is not between 0 and n - 1.
    """
    n = len(vectors)
    if not 0 <= f < n:
        raise ValueError(f"f must be between 0 and {n - 1}, got {f}")
    squared = _squared_distances(vectors)
    best = min(
        combinations(range(n), n - f),
        key=lambda subset: squared[np.ix_(subset, subset)].max(),
    )
    return list(best)


def norm_filter(vectors: Sequence[np.ndarray], bound: float) -> list[int]:
    """Keep the vectors whose L2 norm is at most a bound.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        bound (float): The largest norm kept, e.g. a multiple of the median
            norm.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.
    """
    return [i for i, v in enumerate(vectors) if np.linalg.norm(v) <= bound]


def dnc(
    vectors: Sequence[np.ndarray],
    f: int,
    rng: np.random.Generator,
    iterations: int = 5,
    dimensions: int = 10_000,
    c: float = 1.0,
) -> list[int]:
    """Keep the vectors deviating the least along their spread (Divide-and-Conquer).

    At each iteration, on a random subset of the coordinates, the vectors are
    centred on their mean, and each is scored by its squared projection on the
    top singular vector: the c f highest scores are dropped. A vector is kept
    if no iteration dropped it (Shejwalkar and Houmansadr, 2021).

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.
        rng (np.random.Generator): The source of randomness, e.g. the
            entity's ``np_rng``.
        iterations (int): The number of iterations. Defaults to 5.
        dimensions (int): The number of coordinates sampled at each
            iteration; all of them if there are fewer. Defaults to 10000.
        c (float): The multiple of f dropped at each iteration. Defaults to
            1.0.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.

    Raises:
        ValueError: If ``f`` or ``c`` is negative, ``iterations`` is not
            positive, or c f is not below n.
    """
    stacked = np.vstack(vectors)
    n, d = stacked.shape
    dropped = int(c * f)
    if f < 0 or c < 0 or iterations < 1 or dropped >= n:
        raise ValueError(
            f"DnC requires f, c >= 0, iterations >= 1 and c f < n, got "
            f"f={f}, c={c}, iterations={iterations}, n={n}"
        )
    kept = set(range(n))
    for _ in range(iterations):
        sampled = np.sort(rng.choice(d, size=min(dimensions, d), replace=False))
        centred = stacked[:, sampled] - stacked[:, sampled].mean(axis=0)
        top = np.linalg.svd(centred, full_matrices=False)[2][0]
        scores = (centred @ top) ** 2
        kept &= set(np.argsort(scores, kind="stable")[: n - dropped].tolist())
    return sorted(kept)


def cosine_bipartition(vectors: Sequence[np.ndarray]) -> list[list[int]]:
    """Split the vectors into the two groups least similar to each other.

    The split minimises the largest cosine similarity between two vectors of
    different groups, as clustered federated learning does (Sattler et al.,
    2020): pairs are joined from the most similar down, until two groups are
    left. CFL splits a group when its mean update is small but some updates
    are large, i.e. when its members pull in different directions.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions, at
            least two.

    Returns:
        list[list[int]]: The two groups, each in increasing order, the one
        with the first vector first.

    Raises:
        ValueError: If there are fewer than two vectors.
    """
    n = len(vectors)
    merges = n - 2  # from one group per vector down to two
    if merges < 0:
        raise ValueError(f"a bipartition requires two vectors, got {n}")
    similarity = _cosine_similarities(vectors)
    group = list(range(n))  # the group of each vector, by its smallest member
    pairs = iter(np.argsort(-similarity, axis=None, kind="stable").tolist())
    while merges:
        i, j = divmod(next(pairs), n)
        if group[i] != group[j]:
            old, new = max(group[i], group[j]), min(group[i], group[j])
            group = [new if g == old else g for g in group]
            merges -= 1
    first = [i for i in range(n) if group[i] == 0]
    return [first, [i for i in range(n) if group[i] != 0]]


def bucketing(n: int, size: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle the contributions into buckets (Karimireddy et al., 2022).

    The mean of each bucket is then given to a robust rule instead of the
    contributions themselves, which makes the rule robust on non-IID data too:

    ```python
    means = [aggregation.mean([deltas[i] for i in b]) for b in buckets]
    update = aggregation.median(means)
    ```

    Args:
        n (int): The number of contributions.
        size (int): The size of a bucket; the last one may be smaller.
        rng (np.random.Generator): The source of randomness, e.g. the
            entity's ``np_rng``: a fixed order would defeat the purpose.

    Returns:
        list[list[int]]: The buckets, each a group of positions.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    order = rng.permutation(n).tolist()
    return [sorted(order[i : i + size]) for i in range(0, n, size)]


def nnm(vectors: Sequence[np.ndarray], f: int) -> list[list[int]]:
    """Group each vector with its n - f nearest (nearest-neighbour mixing).

    Each vector is replaced by the mean of its group, itself included, before
    a robust rule (Allouah et al., 2023). The groups overlap.

    Args:
        vectors (Sequence[np.ndarray]): The vectors of the contributions.
        f (int): The number of Byzantine contributions tolerated.

    Returns:
        list[list[int]]: The group of each vector, in the order of the
        vectors.

    Raises:
        ValueError: If ``f`` is not between 0 and n - 1.
    """
    return [monna(vectors, i, f) for i in range(len(vectors))]


def monna(vectors: Sequence[np.ndarray], own: int, f: int) -> list[int]:
    """Keep the n - f vectors nearest to the entity's own (MoNNA).

    A peer averages the vectors it received with its own, keeping the n - f
    nearest to its own, itself included (Farhadkhani et al., 2023).

    Args:
        vectors (Sequence[np.ndarray]): The vectors, the entity's own included.
        own (int): The position of the entity's own vector.
        f (int): The number of Byzantine contributions tolerated.

    Returns:
        list[int]: The positions of the kept vectors, in increasing order.

    Raises:
        ValueError: If ``f`` is not between 0 and n - 1.
    """
    n = len(vectors)
    if not 0 <= f < n:
        raise ValueError(f"f must be between 0 and {n - 1}, got {f}")
    distances = [float(np.sum((v - vectors[own]) ** 2)) for v in vectors]
    return sorted(np.argsort(distances, kind="stable")[: n - f].tolist())


def _squared_distances(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return the squared distance between every two vectors.

    Args:
        vectors (Sequence[np.ndarray]): The vectors.

    Returns:
        np.ndarray: The n x n matrix of squared distances.
    """
    stacked = np.vstack(vectors)
    return ((stacked[:, None, :] - stacked[None, :, :]) ** 2).sum(axis=-1)


def _krum_scores(squared: np.ndarray, f: int) -> np.ndarray:
    """Return the Krum score of each vector, given their squared distances.

    Args:
        squared (np.ndarray): The n x n matrix of squared distances.
        f (int): The number of Byzantine contributions tolerated.

    Returns:
        np.ndarray: The sum of the squared distances of each vector to its
        n - f - 2 nearest others.
    """
    n = len(squared)
    # the first column of each sorted row is the distance of a vector to itself
    return np.sort(squared, axis=1)[:, 1 : max(n - f - 1, 1)].sum(axis=1)


def _cosine_similarities(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return the cosine similarity between every two vectors.

    Args:
        vectors (Sequence[np.ndarray]): The vectors.

    Returns:
        np.ndarray: The n x n matrix of similarities, 0 for a zero vector.
    """
    stacked = np.vstack(vectors).astype(float)
    norms = np.linalg.norm(stacked, axis=1)
    unit = np.divide(
        stacked, norms[:, None], out=np.zeros_like(stacked), where=norms[:, None] > 0
    )
    return unit @ unit.T
