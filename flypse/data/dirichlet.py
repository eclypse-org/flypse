"""Module for the DirichletDataset class."""

from __future__ import annotations

import random
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from flypse.data.dataset import FederatedDataset

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    from flypse.data.dataset import DataSource


class DirichletDataset(FederatedDataset):
    """A federated dataset with Dirichlet-distributed label skew.

    For each class, the proportions of the class items assigned to the
    entities are drawn from a Dirichlet distribution of concentration
    ``alpha``: lower values yield more heterogeneous (non-IID) entities,
    while large values approach the uniform, i.i.d. split. The class of an
    item is obtained, whenever needed, by applying ``class_fn`` to the
    result of ``source[index]``.
    """

    def __init__(
        self,
        source: DataSource,
        class_fn: Callable[[Any], Hashable],
        alpha: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            source (DataSource): The data source of the federation.
            class_fn (Callable[[Any], Hashable]): A function returning the
                class of an item; it is applied to the item returned by
                the source ``__getitem__``.
            alpha (float): The concentration of the Dirichlet distribution.
                Defaults to 1.0.
            seed (int | None): The seed of the split randomness. Defaults
                to None.

        Raises:
            ValueError: If alpha is not positive.
        """
        super().__init__(source)
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.class_fn = class_fn
        self.alpha = alpha
        self.seed = seed

    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        """Assign the items of each class via Dirichlet-drawn proportions.

        Args:
            entity_ids (Sequence[str]): The validated identifiers of the
                entities participating in the split.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity,
            keyed by entity identifier.
        """
        rng = random.Random(self.seed)

        by_class: dict[Hashable, list[int]] = {}
        for index in range(len(self.source)):
            item_class = self.class_fn(self.source[index])
            by_class.setdefault(item_class, []).append(index)

        k = len(entity_ids)
        splits: dict[str, list[int]] = {e: [] for e in entity_ids}
        for indices in by_class.values():
            rng.shuffle(indices)
            allocation = _allocate(_proportions(rng, self.alpha, k), len(indices))

            start = 0
            for j, entity_id in enumerate(entity_ids):
                splits[entity_id].extend(indices[start : start + allocation[j]])
                start += allocation[j]
        return cast("dict[str, Sequence[int]]", splits)


def _proportions(rng: random.Random, alpha: float, k: int) -> list[float]:
    """Draw k proportions from a symmetric Dirichlet distribution.

    Args:
        rng (random.Random): The random number generator.
        alpha (float): The concentration of the distribution.
        k (int): The number of proportions.

    Returns:
        list[float]: The proportions, summing to 1.
    """
    gammas = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(gammas)
    return [1.0 / k] * k if total == 0.0 else [g / total for g in gammas]


def _allocate(proportions: list[float], n: int) -> list[int]:
    """Allocate n items by proportions, with the largest remainder method.

    Args:
        proportions (list[float]): The proportions, summing to 1.
        n (int): The number of items to allocate.

    Returns:
        list[int]: The number of items of each proportion, summing to n.
    """
    allocation = [int(p * n) for p in proportions]
    by_remainder = sorted(
        range(len(proportions)),
        key=lambda j: proportions[j] * n - allocation[j],
        reverse=True,
    )
    for j in range(n - sum(allocation)):
        allocation[by_remainder[j]] += 1
    return allocation
