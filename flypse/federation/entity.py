"""Module for the FederatedEntity class.

An entity is the basic unit of a federation. At this level of abstraction,
an entity is characterized only by the device of the infrastructure it is
deployed onto -- whose management is delegated to ECLYPSE. Any other aspect
of a federated entity (where its data comes from, what it does with it, what
it exchanges, how it collaborates) is to be built incrementally on top of
this foundation.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

from eclypse.remote.service import Service

if TYPE_CHECKING:
    from flypse.data.view import DataView
    from flypse.system import Device


class Entity(Service):
    """A federated entity, characterized only by the device it runs on."""

    def __init__(
        self,
        entity_id: str,
        store_step: bool = False,
    ):
        """Initialize the federated entity.

        Args:
            entity_id (str): The identifier of the entity.
            store_step (bool): Whether the underlying service stores the
                results of each step. Defaults to False.
        """
        super().__init__(
            entity_id,
            communication_interface="mpi",
            store_step=store_step,
        )
        self._data_view: DataView | None = None

    def on_deploy(self):
        """Bind the view of the dataset held by the hosting device."""
        data = cast("Device", self.node).data
        if data is not None:
            self._data_view = data.bind(self.id)

    def on_undeploy(self):
        """Release the view of the dataset bound at deploy time."""
        self._data_view = None

    @property
    def data(self) -> DataView | None:
        """Get the view of the dataset bound to the entity.

        Returns:
            DataView | None: The view of the dataset, or None if the entity
                is not deployed, or the hosting device has no dataset.
        """
        return self._data_view

    async def step(self):
        """Do nothing.

        The entity has no behavior yet: it only holds its view of the data.
        """
        return None
