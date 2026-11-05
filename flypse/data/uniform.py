"""Module for the UniformDataset class."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

from flypse.data.dataset import FederatedDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flypse.data.dataset import DataSource


class UniformDataset(FederatedDataset):
    """A federated dataset splitting the source into uniform blocks.

    The source is partitioned into contiguous, equally-sized blocks (up to
    a remainder of one item, distributed to the first entities).
    """

    def __init__(self, source: DataSource, seed: int | None = None):
        """Initialize the dataset.

        Args:
            source (DataSource): The data source of the federation.
            seed (int | None): Kept for API consistency with the other
                split datasets; the uniform split is deterministic.
                Defaults to None.
        """
        super().__init__(source)
        self.seed = seed
        """The seed given at construction, unused by the split."""

    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        """Compute the contiguous block of source indices of each entity.

        Args:
            entity_ids (Sequence[str]): The validated identifiers of the
                entities participating in the split.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity,
            keyed by entity identifier.
        """
        n = len(self.source)
        k = len(entity_ids)
        base, remainder = divmod(n, k)
        splits: dict[str, list[int]] = {}
        start = 0
        for i, entity_id in enumerate(entity_ids):
            size = base + (1 if i < remainder else 0)
            splits[entity_id] = list(range(start, start + size))
            start += size
        return cast("dict[str, Sequence[int]]", splits)
