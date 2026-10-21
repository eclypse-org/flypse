"""Tests for the Entity class."""

from __future__ import annotations

import numpy as np

from flypse.data import UniformDataset
from flypse.federation import (
    Entity,
    Message,
    ModelVersion,
)
from flypse.system import Device


class FakeMPI:
    """An in-memory stand-in for the ECLYPSE MPI interface."""

    def __init__(self, inbox=None, neighbors=None):
        self.inbox = list(inbox or [])
        self.neighbors = neighbors or []
        self.sent: list[tuple] = []

    async def get_neighbors(self):
        return self.neighbors

    async def send(self, recipients, body):
        self.sent.append(("send", recipients, body))

    async def bcast(self, body):
        self.sent.append(("bcast", None, body))

    async def recv(self):
        return self.inbox.pop(0)


def _deploy(entity: Entity, data=None) -> Entity:
    """Attach the entity to a device holding the given data and deploy it."""
    device = Device.__new__(Device)
    device.build(data=data)
    entity.attach_node(device)
    entity.on_deploy()
    return entity


def _with_mpi(mpi: FakeMPI) -> Entity:
    """Deploy an entity whose MPI interface is the given fake."""
    entity = _deploy(Entity("a"))
    entity._comm = mpi
    return entity


def _split_dataset() -> UniformDataset:
    """Return a 6-item dataset split across two entities."""
    dataset = UniformDataset(list(range(6)))
    dataset._split(["a", "b"])
    return dataset


def test_entity_has_no_data_before_deploy():
    """An entity that is not deployed holds no data."""
    assert Entity("a").data is None


async def test_entity_step_has_no_effect():
    """The entity step is a no-op."""
    assert await Entity("a").step() is None


def test_entity_deploy_binds_its_view():
    """Deploying binds the entity's own portion of the device dataset."""
    entity = _deploy(Entity("b"), _split_dataset())

    assert [entity.data[i] for i in range(len(entity.data))] == [3, 4, 5]


def test_entity_deploy_without_device_data():
    """Deploying on a device without data leaves the entity without a view."""
    entity = _deploy(Entity("a"), None)

    assert entity.data is None


def test_entity_undeploy_releases_the_view():
    """Undeploying drops the bound view."""
    entity = _deploy(Entity("b"), _split_dataset())

    entity.on_undeploy()

    assert entity.data is None


def test_entity_redeploy_rebinds():
    """An undeployed entity can be deployed again."""
    entity = _deploy(Entity("a"), _split_dataset())
    entity.on_undeploy()

    entity.on_deploy()

    assert [entity.data[i] for i in range(len(entity.data))] == [0, 1, 2]


async def test_entity_get_neighbors():
    """The neighbors are those of the underlying interface."""
    entity = _with_mpi(FakeMPI(neighbors=["b", "c"]))

    assert await entity.get_neighbors() == ["b", "c"]


async def test_entity_send_and_bcast_wrap_the_record_and_data():
    """Records travel with the attached data in the message body."""
    mpi = FakeMPI()
    entity = _with_mpi(mpi)
    record = ModelVersion((0, 1), 1.0, np.ones(2))

    await entity.send("b", record, n_samples=3)
    await entity.bcast(record)

    assert mpi.sent == [
        ("send", "b", {"record": record, "data": {"n_samples": 3}}),
        ("bcast", None, {"record": record, "data": {}}),
    ]


async def test_entity_receive_returns_a_message():
    """A received body becomes a message from its sender."""
    record = ModelVersion((1, 0), 2.0, np.zeros(2))
    entity = _with_mpi(
        FakeMPI(inbox=[{"record": record, "data": {"loss": 0.5}, "sender_id": "b"}])
    )

    assert await entity.receive() == Message("b", record, {"loss": 0.5})
