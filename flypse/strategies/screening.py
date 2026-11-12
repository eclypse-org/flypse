"""Module for the screening strategies.

Screening decides which of the received contributions to use, and in which
groups. A filter keeps some and drops the others: e.g., a norm filter drops
suspiciously large updates, Krum keeps the contributions closest to the rest.
Clustering splits them into groups, each combined on its own and sent back to
its members, as in clustered federated learning (CFL). Either way, it is a
partition of the contributions.

A filter returns the positions of the contributions it keeps, in increasing
order, so that their vectors, senders and weights stay aligned:

.. code-block:: python

    kept = krum(deltas, f=1, m=len(deltas) - 1)
    update = aggregation.mean([deltas[i] for i in kept])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def krum(vectors: Sequence[np.ndarray], f: int, m: int = 1) -> list[int]:
    """Keep the m vectors closest to their neighbours.

    The score of a vector is the sum of its squared distances to its n - f - 2
    nearest other vectors; the m vectors with the lowest scores are kept. Krum
    keeps one; Multi-Krum keeps m and averages them.

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
    stacked = np.vstack(vectors)
    squared = ((stacked[:, None, :] - stacked[None, :, :]) ** 2).sum(axis=-1)
    # the first column of each sorted row is the distance of a vector to itself
    scores = np.sort(squared, axis=1)[:, 1 : n - f - 1].sum(axis=1)
    return sorted(np.argsort(scores, kind="stable")[:m].tolist())
