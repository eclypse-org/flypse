"""Module for the aggregation strategies.

Aggregation combines the contributions an entity uses into one vector, with an
aggregation rule: e.g., the weighted mean, the coordinate-wise median, the
trimmed mean. The vector of a contribution is the model it carries (e.g.,
``message.record.parameters``) or its difference from the model it was
generated from (see :meth:`~flypse.federation.model.FederatedModel.base`).

It is apart from :mod:`~flypse.strategies.weighting` because the same weights
can feed different rules, and the robust rules (:func:`median`,
:func:`trimmed_mean`) ignore weights on purpose: weighting would let one
contribution outweigh the others, which is what they guard against.

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
