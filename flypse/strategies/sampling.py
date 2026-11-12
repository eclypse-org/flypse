"""Module for the sampling strategies.

Sampling is choosing which entities to engage, e.g. which clients train in a
round. Not being chosen means not being contacted: no message, and no state on
the entity. The candidates usually are the neighbours of the entity
(``await self.get_neighbors()``), and the randomness its own ``rng``, seeded by
the process.

Each strategy also returns the probability of each candidate to be chosen: an
unbiased aggregate weighs a contribution by p_i / π_i, and privacy accounting
needs the actual scheme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from collections.abc import Collection


def uniform(
    candidates: Collection[str], k: int, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Choose k of the candidates, uniformly and without replacement.

    Args:
        candidates (Collection[str]): The identifiers of the candidates.
        k (int): How many to choose; all of them if there are fewer.
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The chosen candidates, and the
        probability of each candidate to be chosen, k / n.

    Raises:
        ValueError: If ``k`` is negative.

    Example:
        .. code-block:: python

            chosen, _ = uniform(await self.get_neighbors(), 10, self.rng)
            await self.send(chosen, self.model.versions[-1])
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    # sorted, so that the choice depends on the seed and not on the order given
    pool = sorted(candidates)
    k = min(k, len(pool))
    return rng.sample(pool, k), {c: k / len(pool) for c in pool}


def poisson(
    candidates: Collection[str], q: float, rng: random.Random
) -> tuple[list[str], dict[str, float]]:
    """Choose each candidate independently, with probability q.

    The number of chosen candidates varies, and can be 0.

    Args:
        candidates (Collection[str]): The identifiers of the candidates.
        q (float): The probability of each candidate to be chosen, in [0, 1].
        rng (random.Random): The source of randomness, e.g. the entity's
            ``rng``.

    Returns:
        tuple[list[str], dict[str, float]]: The chosen candidates, and the
        probability of each candidate to be chosen, q.

    Raises:
        ValueError: If ``q`` is not in [0, 1].
    """
    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    pool = sorted(candidates)
    return [c for c in pool if rng.random() < q], dict.fromkeys(pool, q)
