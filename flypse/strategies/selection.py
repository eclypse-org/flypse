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

if TYPE_CHECKING:
    import random
    from collections.abc import Collection


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
