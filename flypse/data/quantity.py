"""Module for the QuantitySkewDataset class."""

from __future__ import annotations

import random
from typing import (
    TYPE_CHECKING,
    cast,
)

from flypse.data.dataset import FederatedDataset
from flypse.data.dirichlet import (
    _allocate,
    _proportions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flypse.data.dataset import DataSource


class QuantitySkewDataset(FederatedDataset):
    """A federated dataset with Dirichlet-distributed quantity skew.

    The number of items of each entity is drawn from a Dirichlet distribution
    of concentration ``alpha``: lower values yield more unbalanced entities,
    while large values approach equally-sized entities. Every entity gets at
    least ``min_size`` items, and the items are assigned at random, so the
    entities differ in how much data they hold, not in its distribution.
    """

    def __init__(
        self,
        source: DataSource,
        alpha: float = 1.0,
        min_size: int = 1,
        seed: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            source (DataSource): The data source of the federation.
            alpha (float): The concentration of the Dirichlet distribution.
                Defaults to 1.0.
            min_size (int): The minimum number of items of each entity.
                Defaults to 1.
            seed (int | None): The seed of the split randomness. Defaults
                to None.

        Raises:
            ValueError: If alpha is not positive, or min_size is negative.
        """
        super().__init__(source)
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if min_size < 0:
            raise ValueError("min_size must be non-negative.")
        self.alpha = alpha
        self.min_size = min_size
        self.seed = seed

    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        """Assign each entity a Dirichlet-drawn number of random items.

        Args:
            entity_ids (Sequence[str]): The validated identifiers of the
                entities participating in the split.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity,
            keyed by entity identifier.

        Raises:
            ValueError: If the source cannot give ``min_size`` items to
                every entity.
        """
        n = len(self.source)
        k = len(entity_ids)
        free = n - k * self.min_size
        if free < 0:
            raise ValueError(
                f"{n} items cannot give {self.min_size} to each of {k} entities."
            )
        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        allocation = _allocate(_proportions(rng, self.alpha, k), free)

        splits: dict[str, list[int]] = {}
        start = 0
        for entity_id, extra in zip(entity_ids, allocation, strict=True):
            size = self.min_size + extra
            splits[entity_id] = indices[start : start + size]
            start += size
        return cast("dict[str, Sequence[int]]", splits)
