"""Module for the Device class.

A device is the node of the infrastructure an entity is deployed onto. In a
real federation the data lives on the device, not on the federated process:
every device is given the dataset of the federation, and each entity running
on it binds its own view of that dataset.

The entities deployed onto a device also share its slots, through its
``concurrency``: a device may admit only a given number of them at once in their
critical operations, such as a local training.

A message reaching an entity on the device is stamped with when it arrived and
with its size, before it waits for the entity to receive it.
"""

from __future__ import annotations

from time import time
from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.remote.communication.mpi import EclypseMPI
from eclypse.remote.node import RemoteNode
from ray import cloudpickle

from flypse.system.concurrency import ConcurrencyHandler

if TYPE_CHECKING:
    from pickle import PickleBuffer

    from eclypse.remote.communication import Route

    from flypse.data.dataset import FederatedDataset


class Device(RemoteNode):
    """A node of the infrastructure holding the federation dataset."""

    def build(
        self,
        data: FederatedDataset | None = None,
        max_concurrency: int | None = None,
    ):
        """Store the federation dataset on the device, and set up its slots.

        Args:
            data (FederatedDataset | None): The dataset of the federation.
                Defaults to None.
            max_concurrency (int | None): The number of entities holding a
                slot at once. Defaults to None, meaning no limit.
        """
        super().build()
        self.data = data
        """The dataset of the federation."""
        self.concurrency = ConcurrencyHandler(max_concurrency)
        """The slots shared by the entities on the device."""

    async def service_comm_entrypoint(
        self, route: Route, communication_interface: type, **handle_args: Any
    ) -> Any:
        """Deliver a message to an entity on the device, stamping it first.

        Args:
            route (Route): The route of the message.
            communication_interface (type): The interface delivering it.
            **handle_args: The body of the message.

        Returns:
            Any: The response of the interface.
        """
        if communication_interface is EclypseMPI:
            size = sum(_serialized_size(value) for value in handle_args.values())
            handle_args.update(received_at=time(), size=size)
        return await super().service_comm_entrypoint(
            route, communication_interface, **handle_args
        )


def _serialized_size(value: Any) -> int:
    """Return the size of a value serialized as for the wire, in bytes.

    The arrays are measured without being copied.

    Args:
        value (Any): The value, as delivered.

    Returns:
        int: The size, in bytes.
    """
    buffers: list[PickleBuffer] = []
    inband = cloudpickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    return len(inband) + sum(buffer.raw().nbytes for buffer in buffers)
