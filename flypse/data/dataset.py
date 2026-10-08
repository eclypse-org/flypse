"""Module for the FederatedDataset class.

A :class:`FederatedDataset` embodies the data of the **whole federation**
together with the way it is partitioned across entities. It is constructed
from a ``DataSource`` (anything with ``__len__`` and ``__getitem__``); the
entities of the split are given at split time, either as their count or as
the explicit list of their identifiers. How indices are assigned to
entities is defined by subclasses (e.g.,
:class:`~flypse.data.uniform.UniformDataset`).

The same object can be shared by all the entities of a federation: after
the dataset is split for the federation roster, each entity **binds** it
(``bind``) and gets back a :class:`~flypse.data.view.DataView` addressing
only its own portion. Binding does not mutate the dataset, so several
entities can bind the same shared object at the same time.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

from flypse.data.view import DataView

if TYPE_CHECKING:
    from collections.abc import Sequence


class DataSource(Protocol):
    """A protocol for anything that can serve data items by index.

    Any object exposing ``__len__`` and ``__getitem__`` (a list, an array, a
    dataframe, a torch-style dataset, ...) is a valid ``DataSource``. The
    items it yields are opaque to FLYPSE.
    """

    def __len__(self) -> int:
        """Return the total number of available items."""

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the given index.
        """


class FederatedDataset(ABC):
    """The data of a federation, partitioned across its entities."""

    def __init__(self, source: DataSource):
        """Initialize the federated dataset.

        Args:
            source (DataSource): The data source of the federation.
        """
        self.source = source
        self._entity_ids: list[str] | None = None
        self._splits: dict[str, Sequence[int]] | None = None

    def _split(self, entities: int | Sequence[str]) -> dict[str, Sequence[int]]:
        """Split the source across the given entities, caching the result.

        The split is computed once per entity roster and reused while the
        roster does not change.

        Args:
            entities (int | Sequence[str]): The entities participating in
                the split, either as their count (identifiers "0", ...,
                "n-1") or as the explicit list of their identifiers.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity,
            keyed by entity identifier.

        Raises:
            ValueError: If the entities are empty or duplicated.
        """
        entity_ids = _normalize_entities(entities)
        if self._splits is None or self._entity_ids != entity_ids:
            self._entity_ids = entity_ids
            self._splits = self.split(entity_ids)
        return self._splits

    @abstractmethod
    def split(self, entity_ids: Sequence[str]) -> dict[str, Sequence[int]]:
        """Compute the indices of the source assigned to each entity.

        Args:
            entity_ids (Sequence[str]): The validated identifiers of the
                entities participating in the split.

        Returns:
            dict[str, Sequence[int]]: The source indices of each entity,
            keyed by entity identifier.
        """

    def bind(self, entity_id: str) -> DataView:
        """Bind the dataset to an entity, returning its view.

        Binding does not mutate the dataset: the returned view addresses
        only the portion of the data assigned to the entity, while the
        dataset can be bound by several entities at once.

        Args:
            entity_id (str): The identifier of the binding entity.

        Returns:
            DataView: The view of the data assigned to the entity.

        Raises:
            RuntimeError: If the dataset has not been split yet.
            KeyError: If the entity is not part of the split.
        """
        if self._entity_ids is None or self._splits is None:
            raise RuntimeError("The dataset has not been split yet.")
        if entity_id not in self._entity_ids:
            raise KeyError(f"Unknown entity: {entity_id}.")
        return DataView(self.source, self._splits[entity_id])


def _normalize_entities(entities: int | Sequence[str]) -> list[str]:
    """Normalize and validate the entities of a split.

    Args:
        entities (int | Sequence[str]): The entities, either as their count
            or as the explicit list of their identifiers.

    Returns:
        list[str]: The validated entity identifiers.

    Raises:
        ValueError: If the entities are empty or duplicated.
    """
    if isinstance(entities, int):
        if entities <= 0:
            raise ValueError("The number of entities must be positive.")
        return [str(i) for i in range(entities)]
    entity_ids = [str(e) for e in entities]
    if not entity_ids:
        raise ValueError("At least one entity is required.")
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("Entity identifiers must be unique.")
    return entity_ids
