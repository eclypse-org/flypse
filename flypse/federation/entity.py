"""Module for the Entity class.

An entity is the basic unit of a federation: an ECLYPSE service deployed onto
a device of the infrastructure, from which it binds its view of the federation
data, available as ``data`` (None if the entity holds no data). An entity may
reassign ``data``, e.g. to a part of its view. What an entity does with its
data, what it exchanges, and how it collaborates with the other entities is
defined by subclasses, implementing the ``step`` method.

Entities exchange versions of their models, or control messages carrying only
data, with ``send``, ``bcast`` and ``receive``, and find the entities they are
connected to with ``get_neighbors``, without dealing with the underlying
ECLYPSE MPI interface. Every exchange is counted in the ``traffic`` of the
entity. For its own randomness, an entity draws from its ``rng``, seeded from
the seed of the process and the identifier of the entity, so that entities
draw different but reproducible sequences.

An entity evaluates itself in ``evaluate``, which the federated process calls
on its ``evaluate`` event, keeping the latest results in ``eval_metrics``.
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from eclypse.remote.service import Service

from flypse.federation.message import Message
from flypse.federation.traffic import Traffic
from flypse.seed import get_seed

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
        with_data: bool = True,
    ):
        """Initialize the entity.

        Args:
            entity_id (str): The identifier of the entity.
            store_step (bool): Whether the underlying service stores the
                results of each step. Defaults to False.
            with_data (bool): Whether the entity holds a portion of the
                federation data (e.g., False for an aggregator). Defaults to
                True.
        """
        super().__init__(
            entity_id,
            communication_interface="mpi",
            store_step=store_step,
        )
        self.with_data = with_data
        self.traffic = Traffic()
        self.rng = random.Random()
        self.eval_metrics: dict[str, float] = {}
        self.data: DataView | None = None

    def on_deploy(self):
        """Seed the ``rng`` of the entity and bind its view of the data.

        The ``rng`` is seeded from the seed of the process and the identifier
        of the entity; the view is the one of the dataset held by the hosting
        device.
        """
        seed = get_seed()
        if seed is not None:
            self.rng.seed(f"{seed}/{self.id}")
        data = cast("Device", self.node).data
        if self.with_data and data is not None:
            self.data = data.bind(self.id)

    def on_undeploy(self):
        """Release the view of the dataset bound at deploy time."""
        self.data = None

    async def get_neighbors(self) -> list[str]:
        """Get the entities connected to this entity.

        Returns:
            list[str]: The identifiers of the neighbor entities.
        """
        return await self.mpi.get_neighbors()

    async def send(
        self,
        recipients: str | list[str],
        record: ModelVersion | None = None,
        **data: Any,
    ):
        """Send a message to one or more entities.

        It waits for the modelled communication cost of the message, and
        counts the message in the traffic of the entity.

        Args:
            recipients (str | list[str]): The identifier of the recipient,
                or the list of identifiers of the recipients.
            record (ModelVersion | None): The version of the model to send.
                Defaults to None, meaning a control message carrying only
                its data.
            **data: The data attached to the message.
        """
        start = perf_counter()
        request = await self.mpi.send(recipients, {"record": record, "data": data})
        self._count_sent(request.recipient_ids, record, start)

    async def bcast(self, record: ModelVersion | None = None, **data: Any):
        """Send a message to all the neighbor entities.

        It waits for the modelled communication cost of the message, and
        counts the message in the traffic of the entity.

        Args:
            record (ModelVersion | None): The version of the model to send.
                Defaults to None, meaning a control message carrying only
                its data.
            **data: The data attached to the message.
        """
        start = perf_counter()
        request = await self.mpi.bcast({"record": record, "data": data})
        self._count_sent(request.recipient_ids, record, start)

    async def receive(self, timeout: float | None = None) -> Message:
        """Receive the next message sent to the entity.

        It waits until a message is available, or until the timeout expires.
        A message arriving after the timeout is kept for the next call. The
        message, and the time waited for it, are counted in the traffic of
        the entity.

        Args:
            timeout (float | None): The maximum time to wait, in seconds.
                Defaults to None, meaning no limit.

        Returns:
            Message: The received message.

        Raises:
            TimeoutError: If no message arrives within the timeout.
        """
        start = perf_counter()
        try:
            body = await self.mpi.recv(timeout=timeout)
        finally:
            self.traffic.receive_time += perf_counter() - start
        self.traffic.messages_received += 1
        self.traffic.bytes_received += _nbytes(body["record"])
        return Message(body["sender_id"], body["record"], body["data"])

    def _count_sent(
        self, recipients: list[str], record: ModelVersion | None, start: float
    ):
        """Count a message delivered to the recipients in the traffic.

        Args:
            recipients (list[str]): The identifiers of the recipients.
            record (ModelVersion | None): The version of the model sent.
            start (float): When the sending started, as a ``perf_counter``.
        """
        self.traffic.messages_sent += len(recipients)
        self.traffic.bytes_sent += len(recipients) * _nbytes(record)
        self.traffic.send_time += perf_counter() - start

    def evaluate(self) -> dict[str, float] | None:
        """Evaluate the entity, e.g. its model on its test data.

        It is called on the device while the entity keeps stepping, so it
        should evaluate a snapshot of the model (e.g. a ``ModelVersion``
        record) rather than the live one. It does nothing by default, and is
        meant to be overridden by subclasses.

        Returns:
            dict[str, float] | None: The value of each evaluation metric,
            keyed by name, or None if the entity does not evaluate.
        """
        return None

    async def step(self):
        """Run a step of the entity.

        It does nothing by default, and is meant to be overridden by
        subclasses.
        """
        return None


def _nbytes(record: ModelVersion | None) -> int:
    """Return the bytes of model carried by a message.

    Args:
        record (ModelVersion | None): The version of the model, if any.

    Returns:
        int: The bytes of the parameters, or 0 for a control message.
    """
    return 0 if record is None else record.parameters.nbytes
