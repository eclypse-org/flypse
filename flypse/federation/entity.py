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
ECLYPSE MPI interface. For its own randomness, an entity draws from its ``rng``,
seeded from the seed of the process and the identifier of the entity, so that
entities draw different but reproducible sequences.

An entity pushes whatever it wants to report straight to the simulator with
``report``, instead of waiting for a metric to read it. It evaluates a version of
a model by calling ``run_evaluation`` from its ``step``: the ``evaluate`` hook
computes the results on the data or on one of the named ``splits``, and they are
reported at once, identifying the round, the version and the parameters of the
evaluated model, and the split.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from hashlib import blake2b
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from eclypse.remote.service import Service

from flypse.federation.message import Message
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
        self.rng = random.Random()
        self.n_reports = 0
        self.data: DataView | None = None
        self.splits: dict[str, DataView] = {}

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
        """Release the view of the dataset bound at deploy time, and its splits."""
        self.data = None
        self.splits = {}

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

        It waits for the modelled communication cost of the message.

        Args:
            recipients (str | list[str]): The identifier of the recipient,
                or the list of identifiers of the recipients.
            record (ModelVersion | None): The version of the model to send.
                Defaults to None, meaning a control message carrying only
                its data.
            **data: The data attached to the message.
        """
        await self.mpi.send(recipients, {"record": record, "data": data})

    async def bcast(self, record: ModelVersion | None = None, **data: Any):
        """Send a message to all the neighbor entities.

        It waits for the modelled communication cost of the message.

        Args:
            record (ModelVersion | None): The version of the model to send.
                Defaults to None, meaning a control message carrying only
                its data.
            **data: The data attached to the message.
        """
        await self.mpi.bcast({"record": record, "data": data})

    async def receive(self, timeout: float | None = None) -> Message:
        """Receive the next message sent to the entity.

        It waits until a message is available, or until the timeout expires.
        A message arriving after the timeout is kept for the next call.

        Args:
            timeout (float | None): The maximum time to wait, in seconds.
                Defaults to None, meaning no limit.

        Returns:
            Message: The received message.

        Raises:
            TimeoutError: If no message arrives within the timeout.
        """
        body = await self.mpi.recv(timeout=timeout)
        return Message(body["sender_id"], body["record"], body["data"])

    def report(self, **values: Any):
        """Push values to the simulator, which writes them in the report.

        It does not wait for the simulator, and it is meant to be called from
        ``step``, as often as a round allows. The values are reported together,
        with the name of the entity and of its device, when they were reported,
        and the progressive number of the report: ``entity_id``, ``node_id``,
        ``timestamp`` and ``n_report`` are therefore reserved names.

        Args:
            **values: The values to report, by name. They can be anything JSON
                can represent, and arrays or scalars of the numerical libraries.

        Raises:
            TypeError: If a value cannot be represented.
        """
        # eclypse#51: a value the writer cannot serialize would hang the shutdown
        values = json.loads(json.dumps(values, default=_to_plain))
        self.node.get_actor(self.node.manager_actor_name).report.remote(
            self.application_id,
            self.id,
            self.n_reports + 1,
            datetime.now().isoformat(),
            {**values, "node_id": self.node.id},
        )
        self.n_reports += 1

    def evaluate(
        self,
        record: ModelVersion,
        data: DataView | None,
    ) -> dict[str, float] | None:
        """Evaluate a version of a model on the given data.

        It is called by ``run_evaluation``, and the version need not be the
        loaded one: read its parameters with ``get_parameters(version=record)``.
        It does nothing by default, and is meant to be overridden by subclasses.

        Args:
            record (ModelVersion): The version of the model to evaluate.
            data (DataView | None): The data to evaluate it on, or None if the
                entity holds no data.

        Returns:
            dict[str, float] | None: The value of each evaluation metric,
            keyed by name, or None if the entity does not evaluate.
        """

    def run_evaluation(
        self,
        record: ModelVersion,
        split: str | None = None,
        round: int | None = None,
    ):
        """Evaluate a version of a model, and report the results.

        It is meant to be called from ``step``, with any version the entity
        holds: its own, or one it received, loaded or not. The results are
        reported together. The model is identified by ``version`` and by the
        ``model`` hash of its parameters, as a version alone may be shared by
        different models (e.g., two clients generating ``(3, 1)``), while
        entities evaluating the same model get the same hash.

        A failing evaluation is logged and reports nothing, so that the entity
        carries on.

        Args:
            record (ModelVersion): The version of the model to evaluate.
            split (str | None): The name of the part of ``splits`` to evaluate
                it on. Defaults to None, meaning the ``data`` of the entity.
            round (int | None): The round of the evaluation. Defaults to None,
                meaning the coarsest counter of the version.
        """
        try:
            data = self.data if split is None else self.splits[split]
            metrics = self.evaluate(record, data)
            if metrics is None:
                return
            self.report(
                round=record.version[0] if round is None else round,
                version=record.version,
                model=blake2b(record.parameters.tobytes(), digest_size=8).hexdigest(),
                split=split,
                metrics=metrics,
            )
        except Exception:
            self.logger.exception("Evaluation failed.")

    async def step(self):
        """Run a step of the entity.

        It does nothing by default, and is meant to be overridden by
        subclasses.
        """
        return None


def _to_plain(value: Any) -> Any:
    """Convert a value JSON cannot represent, if it is numerical.

    Args:
        value (Any): The value, such as a numpy scalar or array.

    Returns:
        Any: The value as plain numbers.

    Raises:
        TypeError: If the value has no plain form.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot report a value of type {type(value).__name__}.")
