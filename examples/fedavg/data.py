"""Synthetic regression data for the federated averaging example.

The source is sorted by the input feature, so a uniform (contiguous) split
gives every entity a different region of the feature space: the federation is
statistically heterogeneous (non-IID) by construction.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from flypse.data import UniformDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

Item = tuple[list[float], float]
"""An item of the data source: its features and its target."""

TRUE_WEIGHTS = [2.0]
"""The weights of the ground-truth linear model."""

TRUE_BIAS = 1.0
"""The bias of the ground-truth linear model."""


def make_source(n_samples: int, *, noise: float, seed: int) -> list[Item]:
    """Build a synthetic linear-regression data source.

    Args:
        n_samples (int): The number of samples.
        noise (float): The amplitude of the uniform noise added to the targets.
        seed (int): The seed of the noise.

    Returns:
        list[Item]: The data source, sorted by the input feature.
    """
    rng = random.Random(seed)
    source: list[Item] = []
    for i in range(n_samples):
        x = -1.0 + 2.0 * i / max(n_samples - 1, 1)
        y = TRUE_WEIGHTS[0] * x + TRUE_BIAS + rng.uniform(-noise, noise)
        source.append(([x], y))
    return source


class RegressionDataset(UniformDataset):
    """A federated dataset that keeps the aggregator out of the data.

    The aggregator does not hold training data: the whole source is split
    among the learners, and the aggregator gets an empty portion.
    """

    def __init__(
        self,
        source: list[Item],
        server_id: str,
        seed: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            source (list[Item]): The data source of the federation.
            server_id (str): The identifier of the aggregator entity.
            seed (int | None): The seed of the split. Defaults to None.
        """
        super().__init__(source, seed=seed)
        self._server_id = server_id

    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        """Split the source among the learners, leaving the aggregator empty.

        Args:
            entity_ids (Sequence[str]): The validated identifiers of the
                entities participating in the split.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity.
        """
        clients = [e for e in entity_ids if e != self._server_id]
        if not clients:
            return {self._server_id: []}
        splits = dict(super().split(clients))
        splits[self._server_id] = []
        return splits
