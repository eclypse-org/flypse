"""Module for the Entity class.

An entity is the basic unit of a federation: an ECLYPSE service deployed onto
a device of the infrastructure, from which it binds its view of the federation
data. What an entity does with its data, what it exchanges, and how it
collaborates with the other entities is defined by subclasses, implementing
the ``step`` method.

Entities exchange versions of their models with ``send``, ``bcast`` and
``receive``, and find the entities they are connected to with
``get_neighbors``, without dealing with the underlying ECLYPSE MPI interface.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from eclypse.remote.service import Service

from flypse.federation.message import Message

if TYPE_CHECKING:
    from flypse.data.view import DataView
    from flypse.federation.model import ModelVersion
    from flypse.system import Device


class Entity(Service):
    """A federated entity, running on a device and bound to its data."""

    def __init__(
        self,
        entity_id: str,
        store_step: bool = False,
    ):
        """Initialize the entity.

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

    async def get_neighbors(self) -> list[str]:
        """Get the entities connected to this entity.

        Returns:
            list[str]: The identifiers of the neighbor entities.
        """
        return await self.mpi.get_neighbors()

    async def send(
        self, recipients: str | list[str], record: ModelVersion, **data: Any
    ):
        """Send a version of the model to one or more entities.

        It waits for the modelled communication cost of the message.

        Args:
            recipients (str | list[str]): The identifier of the recipient,
                or the list of identifiers of the recipients.
            record (ModelVersion): The version of the model to send.
            **data: The data attached to the message.
        """
        await self.mpi.send(recipients, {"record": record, "data": data})

    async def bcast(self, record: ModelVersion, **data: Any):
        """Send a version of the model to all the neighbor entities.

        It waits for the modelled communication cost of the message.

        Args:
            record (ModelVersion): The version of the model to send.
            **data: The data attached to the message.
        """
        await self.mpi.bcast({"record": record, "data": data})

    async def receive(self) -> Message:
        """Receive the next message sent to the entity.

        It waits until a message is available.

        Returns:
            Message: The received message.
        """
        body = await self.mpi.recv()
        return Message(body["sender_id"], body["record"], body["data"])

    async def step(self):
        """Run a step of the entity.

        It does nothing by default, and is meant to be overridden by
        subclasses.
        """
        return None
