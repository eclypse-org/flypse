"""Tests for the Entity class."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from eclypse.utils.constants import RND_SEED

from flypse.data import (
    FractionSplitter,
    UniformDataset,
)
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

    async def recv(self, timeout=None):
        self.timeout = timeout
        if not self.inbox:
            raise TimeoutError
        return self.inbox.pop(0)


class FakeSimulator:
    """An in-memory stand-in for the simulator actor."""

    def __init__(self):
        self.reports: list[tuple] = []
        self.report = SimpleNamespace(remote=self._receive)

    def _receive(self, *report):
        self.reports.append(report)

    @property
    def values(self) -> list[dict]:
        """The reported values, in order."""
        return [report[-1] for report in self.reports]


def _deploy(entity: Entity, data=None) -> Entity:
    """Attach the entity to a device holding the given data and deploy it."""
    device = Device.__new__(Device)
    device.build(data=data)
    entity.attach_node(device)
    entity.on_deploy()
    return entity


def _connect(entity: Entity) -> FakeSimulator:
    """Attach the entity to a device reaching a fake simulator."""
    simulator = FakeSimulator()
    device = Device.__new__(Device)
    device.build()
    device._node_id = "device"
    device._infrastructure_id = "infrastructure"
    device._actor_cache = {"infrastructure/manager": simulator}
    entity.attach_node(device)
    entity.application_id = "Fed"
    return simulator


class Evaluating(Entity):
    """An entity returning given results, keeping what it was asked to evaluate."""

    def __init__(self, results):
        super().__init__("a")
        self.results = results
        self.evaluated: list[tuple] = []

    def evaluate(self, record, data):
        self.evaluated.append((record, data))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


def _record(version=(0, 0), value=0.0) -> ModelVersion:
    """Return a record of the given version with a single parameter."""
    return ModelVersion(version, 0.0, np.array([value], dtype=np.float32))


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


def test_entity_report_carries_its_identity():
    """A report says who sent it, from where and when, besides its values."""
    entity = Entity("a")
    simulator = _connect(entity)

    entity.report(round=3, split="test")

    ((application_id, entity_id, n_report, timestamp, values),) = simulator.reports
    assert (application_id, entity_id, n_report) == ("Fed", "a", 1)
    assert datetime.fromisoformat(timestamp)
    assert values == {"round": 3, "split": "test", "node_id": "device"}


def test_entity_numbers_its_reports():
    """The reports are numbered, so the simulator knows how many to expect."""
    entity = Entity("a")
    simulator = _connect(entity)

    entity.report(value=1)
    entity.report(value=2)

    assert [report[2] for report in simulator.reports] == [1, 2]
    assert entity.n_reports == 2


def test_entity_reports_plain_numbers():
    """Numerical scalars and arrays are converted, as the writer cannot take them."""
    entity = Entity("a")
    simulator = _connect(entity)

    entity.report(loss=np.float32(0.5), n=np.int64(2), weights=np.array([1.0, 2.0]))

    (values,) = simulator.values
    assert (type(values["loss"]), type(values["n"])) == (float, int)
    assert values["weights"] == [1.0, 2.0]


def test_entity_refuses_values_the_report_cannot_write():
    """Such a value fails where it is reported, instead of stopping the report."""
    entity = Entity("a")
    simulator = _connect(entity)

    with pytest.raises(TypeError, match="object"):
        entity.report(value=object())

    assert not simulator.reports
    assert entity.n_reports == 0


def test_entity_does_not_evaluate_by_default():
    """The entity has no evaluation by default."""
    entity = Entity("a")

    entity.run_evaluation(_record())

    assert entity.n_reports == 0


def test_run_evaluation_reports_one_record():
    """A report identifies the round, the version, the model and the split."""
    entity = Evaluating({"loss": 0.5, "accuracy": 0.9})
    simulator = _connect(entity)
    entity.splits = {"test": "the test part"}
    record = _record((3, 1))

    entity.run_evaluation(record, "test")

    assert entity.evaluated == [(record, "the test part")]
    (values,) = simulator.values
    assert values == {
        "round": 3,
        "version": [3, 1],
        "model": values["model"],
        "split": "test",
        "metrics": {"loss": 0.5, "accuracy": 0.9},
        "node_id": "device",
    }


def test_run_evaluation_defaults_to_the_data_of_the_entity():
    """Without a split, the whole data of the entity is evaluated."""
    entity = Evaluating({"loss": 0.5})
    simulator = _connect(entity)
    entity.data = "the data"

    entity.run_evaluation(_record())

    assert entity.evaluated[0][1] == "the data"
    assert simulator.values[0]["split"] is None


def test_run_evaluation_accepts_an_explicit_round():
    """The round may differ from the coarsest counter of the version."""
    entity = Evaluating({"loss": 0.5})
    simulator = _connect(entity)

    entity.run_evaluation(_record((0, 7)), round=np.int64(7))

    assert simulator.values[0]["round"] == 7
    assert type(simulator.values[0]["round"]) is int


def test_run_evaluation_reports_plain_numbers():
    """Numpy scalars are converted, as the report cannot serialize them."""
    entity = Evaluating({"loss": np.float32(0.5)})
    simulator = _connect(entity)

    entity.run_evaluation(_record())

    assert type(simulator.values[0]["metrics"]["loss"]) is float


def test_run_evaluation_tells_apart_models_sharing_a_version():
    """The hash follows the parameters, whoever evaluates and whatever the version."""
    first, second = Evaluating({"loss": 0.5}), Evaluating({"loss": 0.5})
    reports, others = _connect(first), _connect(second)

    first.run_evaluation(_record((3, 1), 1.0))
    first.run_evaluation(_record((3, 1), 2.0))
    second.run_evaluation(_record((3, 1), 1.0))

    assert reports.values[0]["model"] != reports.values[1]["model"]
    assert reports.values[0]["model"] == others.values[0]["model"]


@pytest.mark.parametrize(
    ("results", "split"),
    [
        (RuntimeError("boom"), None),
        ({"loss": 0.5}, "unknown"),
        ({"loss": object()}, None),
    ],
)
def test_failing_evaluation_is_logged_and_reports_nothing(results, split, monkeypatch):
    """A failure is logged, and the entity carries on."""
    logger = Mock()
    monkeypatch.setattr(Entity, "logger", logger)
    entity = Evaluating(results)

    entity.run_evaluation(_record(), split)

    assert entity.n_reports == 0
    logger.exception.assert_called_once()


def test_entity_deploy_binds_its_view():
    """Deploying binds the entity's own portion of the device dataset."""
    entity = _deploy(Entity("b"), _split_dataset())

    assert [entity.data[i] for i in range(len(entity.data))] == [3, 4, 5]


def test_entity_without_data_does_not_bind():
    """An entity holding no data binds nothing, even on a device with data."""
    entity = _deploy(Entity("a", with_data=False), _split_dataset())

    assert entity.data is None


def test_entity_deploy_without_device_data():
    """Deploying on a device without data leaves the entity without a view."""
    entity = _deploy(Entity("a"), None)

    assert entity.data is None


def test_entity_undeploy_releases_the_view():
    """Undeploying drops the bound view."""
    entity = _deploy(Entity("b"), _split_dataset())
    entity.splits = {"test": entity.data}

    entity.on_undeploy()

    assert entity.data is None
    assert entity.splits == {}


def test_entity_redeploy_rebinds():
    """An undeployed entity can be deployed again."""
    entity = _deploy(Entity("a"), _split_dataset())
    entity.on_undeploy()

    entity.on_deploy()

    assert [entity.data[i] for i in range(len(entity.data))] == [0, 1, 2]


def test_entity_can_reassign_its_data():
    """An entity can replace its view, e.g. with a part of it."""
    entity = _deploy(Entity("b"), _split_dataset())
    parts = FractionSplitter({"train": 2 / 3, "test": 1 / 3}, order=int).apply(
        entity.data
    )

    entity.data = parts["train"]

    assert [entity.data[i] for i in range(len(entity.data))] == [3, 4]


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


async def test_entity_receive_forwards_the_timeout():
    """The timeout bounds the wait of the underlying interface."""
    mpi = FakeMPI()
    entity = _with_mpi(mpi)

    with pytest.raises(TimeoutError):
        await entity.receive(timeout=0.5)

    assert mpi.timeout == 0.5


async def test_entity_control_messages_carry_only_data():
    """A message without a model travels with its data alone."""
    mpi = FakeMPI()
    entity = _with_mpi(mpi)

    await entity.send("b", stop=True)
    await entity.bcast(round=3)

    assert mpi.sent == [
        ("send", "b", {"record": None, "data": {"stop": True}}),
        ("bcast", None, {"record": None, "data": {"round": 3}}),
    ]


def test_entity_rng_is_seeded_per_entity_at_deploy(monkeypatch):
    """Entities draw different, reproducible sequences from the process seed."""
    monkeypatch.setenv(RND_SEED, "7")

    def draws(entity_id: str) -> list[float]:
        entity = _deploy(Entity(entity_id))
        return [entity.rng.random() for _ in range(3)]

    assert draws("a") == draws("a")
    assert draws("a") != draws("b")
