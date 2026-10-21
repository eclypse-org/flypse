"""Tests for the placement strategy builders."""

from __future__ import annotations

import pytest
from eclypse.graph import Infrastructure
from eclypse.graph.assets.defaults import cpu
from eclypse.placement import PlacementView

from flypse.federation import Entity, Federation
from flypse.system import (
    SINGLETON_NODE,
    ExclusiveStrategy,
    get_mirror_strategy,
    get_singleton_strategy,
)


def _federation(*entity_ids: str) -> Federation:
    """Return a federation with one entity per given identifier."""
    federation = Federation("Fed")
    for entity_id in entity_ids:
        federation.add_entity(Entity(entity_id))
    return federation


def test_singleton_strategy_maps_every_entity_to_the_singleton_node():
    """Every entity is statically mapped onto the singleton node."""
    strategy = get_singleton_strategy(_federation("a", "b"))

    assert strategy.mapping == {
        "a": SINGLETON_NODE,
        "b": SINGLETON_NODE,
    }


def test_singleton_strategy_node_can_be_customized():
    """The node hosting every entity can be customized."""
    strategy = get_singleton_strategy(_federation("a"), node="device-1")

    assert strategy.mapping == {"a": "device-1"}


def test_singleton_strategy_requires_entities():
    """Without entities there is no mapping to build."""
    with pytest.raises(ValueError, match="valid mapping"):
        get_singleton_strategy(Federation("Fed"))


def test_mirror_strategy_maps_every_entity_to_itself():
    """Every entity is mapped onto the device with the same identifier."""
    strategy = get_mirror_strategy(_federation("a", "b"))

    assert strategy.mapping == {"a": "a", "b": "b"}


def _cpu_federation(**cpus: int) -> Federation:
    """Return a federation whose entities only require the given cpu."""
    federation = Federation(
        "Fed", node_assets={"cpu": cpu()}, edge_assets={}, include_default_assets=False
    )
    for entity_id, value in cpus.items():
        federation.add_entity(Entity(entity_id), cpu=value)
    return federation


def _cpu_infrastructure(**cpus: int) -> Infrastructure:
    """Return an infrastructure whose devices only offer the given cpu."""
    infrastructure = Infrastructure(
        "Infra",
        node_assets={"cpu": cpu()},
        edge_assets={},
        path_assets_aggregators={},
    )
    for device_id, value in cpus.items():
        infrastructure.add_node(device_id, cpu=value)
    return infrastructure


def _place(federation: Federation, infrastructure: Infrastructure) -> dict:
    """Run the exclusive strategy on the current state of the infrastructure."""
    return ExclusiveStrategy().place(
        infrastructure, federation, {}, PlacementView(infrastructure)
    )


def test_exclusive_places_each_entity_on_its_own_device():
    """Every entity gets a distinct device satisfying it."""
    mapping = _place(_cpu_federation(a=1, b=1), _cpu_infrastructure(x=2, y=2, z=2))

    assert set(mapping) == {"a", "b"}
    assert len(set(mapping.values())) == 2


def test_exclusive_finds_the_assignment_a_greedy_choice_misses():
    """The matching gives the only fitting device to the entity needing it."""
    mapping = _place(_cpu_federation(a=1, b=4), _cpu_infrastructure(x=4, y=1))

    assert mapping == {"a": "y", "b": "x"}


def test_exclusive_places_nothing_without_enough_devices():
    """Fewer devices than entities means no placement."""
    assert _place(_cpu_federation(a=1, b=1), _cpu_infrastructure(x=8)) == {}


def test_exclusive_places_nothing_when_a_requirement_is_unmet():
    """An entity no device can host means no placement."""
    assert _place(_cpu_federation(a=1, b=16), _cpu_infrastructure(x=8, y=8)) == {}
