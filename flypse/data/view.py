"""Module for the DataView class.

A :class:`DataView` is the portion of a data source assigned to a single
entity. It is returned by :meth:`~flypse.data.dataset.FederatedDataset.bind`
and delegates indexing to the source: the ``i``-th item of the view is the
item of the source at the ``i``-th index of the entity's split.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flypse.data.dataset import DataSource


class DataView:
    """The portion of a data source assigned to a single entity.

    It can be indexed, iterated and sliced, and a slice is another view of the
    same source, so nothing is copied.
    """

    def __init__(self, source: DataSource, indices: Sequence[int]):
        """Initialize the view.

        Args:
            source (DataSource): The data source of the federation.
            indices (Sequence[int]): The source indices of the entity.
        """
        self.source = source
        """The source the view delegates to: the data source, or a parent view."""
        self._indices = indices

    def __len__(self) -> int:
        """Return the number of items in the entity's portion.

        Returns:
            int: The number of items.
        """
        return len(self._indices)

    def __getitem__(self, index: int | slice) -> Any:
        """Return the item of the entity's portion at the given index.

        A slice returns a :class:`DataView` of the same source over the sliced
        indices, so nothing is copied.

        Args:
            index (int | slice): The index or slice within the entity's portion.

        Returns:
            Any: The item of the source at the corresponding position, or a
                :class:`DataView` when ``index`` is a slice.
        """
        if isinstance(index, slice):
            return DataView(self.source, self._indices[index])
        return self.source[self._indices[index]]
