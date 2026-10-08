"""Tests for the FederatedProcess class."""

from __future__ import annotations

import pytest
from eclypse.graph import Infrastructure
from eclypse.placement.strategies import (
    RandomStrategy,
    StaticStrategy,
)
from eclypse.remote.bootstrap import RemoteBootstrap
from eclypse.simulation import Simulation

from flypse.data import UniformDataset
from flypse.federation import Entity, Federation
from flypse.process import FederatedProcess
from flypse.system import (
    SINGLETON_NODE,
    Device,
)


@pytest.fixture(autouse=True)
def fake_build(monkeypatch):
    """Replace the Ray bootstrapping with a recording stub."""
    captured: dict[str, object] = {}
    simulator = object()

    def build(self, infrastructure, simulation_config=None):
        captured["bootstrap"] = self
        captured["infrastructure"] = infrastructure
        captured["simulation_config"] = simulation_config
        return simulator

    monkeypatch.setattr(RemoteBootstrap, "build", build)
    # `FederatedProcess.__init__` registers the federation: keep it off Ray.
    monkeypatch.setattr(
        Simulation,
        "register",
        lambda _self, application, placement_strategy=None: None,
    )
    return captured, simulator


def _federation(*entity_ids: str) -> Federation:
    """Return a federation with one entity per given identifier."""
    federation = Federation("Fed")
    for entity_id in entity_ids:
        federation.add_entity(Entity(entity_id))
    return federation


def test_federated_process_is_a_simulation():
    """A federated process is an ECLYPSE simulation."""
    assert issubclass(FederatedProcess, Simulation)


def test_process_wires_devices_and_data(fake_build):
    """The process emulates the federation through devices holding the data."""
    captured, simulator = fake_build
    infrastructure = Infrastructure("Infra")
    federation = _federation("a", "b")
    dataset = UniformDataset(list(range(6)))

    process = FederatedProcess(federation, dataset, infrastructure)

    assert process.federation is federation
    assert process.data is dataset
    assert process.simulator is simulator
    assert captured["infrastructure"] is infrastructure
    assert captured["bootstrap"] is process.remote
    assert process.remote is not None
    assert process.remote._node_class is Device
    assert process.remote.node_args["data"] is dataset


def test_process_splits_the_data_for_the_roster():
    """The data is split for the entities before the devices are created."""
    dataset = UniformDataset(list(range(6)))
    federation = _federation("a", "b")

    FederatedProcess(federation, dataset, Infrastructure("Infra"))

    view_a = dataset.bind("a")
    view_b = dataset.bind("b")
    assert [view_a[i] for i in range(len(view_a))] == [0, 1, 2]
    assert [view_b[i] for i in range(len(view_b))] == [3, 4, 5]


def test_process_without_entities_does_not_split():
    """Without entities the data is left untouched."""
    dataset = UniformDataset(list(range(6)))

    FederatedProcess(Federation("Fed"), dataset, Infrastructure("Infra"))

    with pytest.raises(RuntimeError, match="not been split"):
        dataset.bind("a")


def test_process_forwards_the_simulation_config():
    """The simulation configuration arguments are forwarded to ECLYPSE."""
    process = FederatedProcess(
        _federation("a"),
        UniformDataset(list(range(2))),
        Infrastructure("Infra"),
        seed=7,
        max_steps=3,
    )

    assert process._sim_config.seed == 7
    assert process._sim_config.max_steps == 3
    assert process.remote is not None


def test_register_registers_the_federation_of_the_process(monkeypatch):
    """Registering forwards the federation of the process to ECLYPSE."""
    registered: dict[str, object] = {}

    def fake_register(_self, application, placement_strategy=None):
        registered["application"] = application
        registered["strategy"] = placement_strategy

    monkeypatch.setattr(Simulation, "register", fake_register)
    federation = _federation("a")
    process = FederatedProcess(
        federation,
        UniformDataset(list(range(2))),
        Infrastructure("Infra"),
    )
    strategy = RandomStrategy(seed=1)

    process.register(federation, strategy)

    assert registered["application"] is federation
    assert registered["strategy"] is strategy


def test_register_defaults_to_the_federation_of_the_process(monkeypatch):
    """The federation of the process can be omitted when registering."""
    registered: dict[str, object] = {}

    def fake_register(_self, application, placement_strategy=None):
        registered["application"] = application
        registered["strategy"] = placement_strategy

    monkeypatch.setattr(Simulation, "register", fake_register)
    federation = _federation("a")
    process = FederatedProcess(
        federation,
        UniformDataset(list(range(2))),
        Infrastructure("Infra"),
    )

    process.register()

    assert registered["application"] is federation
    assert registered["strategy"] is None


def test_register_rejects_another_federation():
    """A federated process emulates a single federation."""
    federation = _federation("a")
    other = _federation("b")
    process = FederatedProcess(
        federation,
        UniformDataset(list(range(2))),
        Infrastructure("Infra"),
    )

    with pytest.raises(ValueError, match="single federation"):
        process.register(other, RandomStrategy(seed=1))


def test_singleton_infrastructure_has_a_single_node():
    """The singleton infrastructure is a single node."""
    process = FederatedProcess(
        _federation("a", "b"),
        UniformDataset(list(range(4))),
        "singleton",
    )

    assert process.infrastructure.id == "singleton"
    assert list(process.infrastructure.nodes) == [SINGLETON_NODE]


def test_singleton_maps_every_entity_to_the_single_node():
    """The singleton strategy is a static mapping onto the only node."""
    process = FederatedProcess(
        _federation("a", "b", "c"),
        UniformDataset(list(range(6))),
        "singleton",
    )

    strategy = process._placement_strategy
    assert isinstance(strategy, StaticStrategy)
    assert strategy.mapping == {
        "a": SINGLETON_NODE,
        "b": SINGLETON_NODE,
        "c": SINGLETON_NODE,
    }


def test_singleton_without_entities_has_no_strategy():
    """An empty singleton federation has no placement to compute."""
    process = FederatedProcess(
        Federation("Fed"),
        UniformDataset(list(range(2))),
        "singleton",
    )

    assert process._placement_strategy is None


def test_mirror_mirrors_the_federation():
    """The mirror infrastructure copies the federation topology."""
    federation = _federation("a", "b")
    federation.add_connection("a", "b")

    process = FederatedProcess(
        federation,
        UniformDataset(list(range(4))),
        "mirror",
    )

    assert process.infrastructure.id == "mirror"
    assert set(process.infrastructure.nodes) == {"a", "b"}
    for node in federation.nodes:
        assert dict(process.infrastructure.nodes[node]) == dict(federation.nodes[node])


def test_mirror_maps_every_entity_to_itself():
    """The mirror strategy maps each entity onto its own device."""
    process = FederatedProcess(
        _federation("a", "b"),
        UniformDataset(list(range(4))),
        "mirror",
    )

    strategy = process._placement_strategy
    assert isinstance(strategy, StaticStrategy)
    assert strategy.mapping == {"a": "a", "b": "b"}


def test_register_uses_the_process_strategy(monkeypatch):
    """Registering without a strategy uses the process strategy."""
    registered: dict[str, object] = {}

    def fake_register(_self, application, placement_strategy=None):
        registered["application"] = application
        registered["strategy"] = placement_strategy

    monkeypatch.setattr(Simulation, "register", fake_register)
    process = FederatedProcess(
        _federation("a"),
        UniformDataset(list(range(2))),
        "singleton",
    )

    process.register()

    assert registered["strategy"] is process._placement_strategy


def test_register_prefers_an_explicit_strategy(monkeypatch):
    """An explicit strategy overrides the process strategy."""
    registered: dict[str, object] = {}

    def fake_register(_self, application, placement_strategy=None):
        registered["application"] = application
        registered["strategy"] = placement_strategy

    monkeypatch.setattr(Simulation, "register", fake_register)
    process = FederatedProcess(
        _federation("a"),
        UniformDataset(list(range(2))),
        "singleton",
    )
    strategy = RandomStrategy(seed=1)

    process.register(placement_strategy=strategy)

    assert registered["strategy"] is strategy
