"""Tests for the infrastructure builders."""

from __future__ import annotations

from eclypse.graph.assets.defaults import get_default_path_aggregators

from flypse.federation import (
    Entity,
    Federation,
)
from flypse.system import (
    SINGLETON_NODE,
    get_mirror,
    get_singleton,
)


def _federation(*entity_ids: str) -> Federation:
    """Return a federation with one entity per given identifier."""
    federation = Federation("Fed")
    for entity_id in entity_ids:
        federation.add_entity(Entity(entity_id))
    return federation


def test_singleton_has_a_single_node():
    """The singleton infrastructure is one node with maximal resources."""
    infrastructure = get_singleton()

    assert infrastructure.id == "singleton"
    assert list(infrastructure.nodes) == [SINGLETON_NODE]
    assert set(infrastructure.node_assets) == set(Federation("Fed").node_assets)
    assert infrastructure.attr_init == "max"


def test_singleton_without_default_assets():
    """The default assets can be left out."""
    infrastructure = get_singleton(include_default_assets=False)

    assert list(infrastructure.node_assets) == []


def test_singleton_infrastructure_id_can_be_customized():
    """The identifier of the singleton infrastructure can be customized."""
    infrastructure = get_singleton(infrastructure_id="single")

    assert infrastructure.id == "single"


def test_singleton_node_assets_values_are_applied():
    """The values of the node assets are installed on the only node."""
    infrastructure = get_singleton(
        include_default_assets=True,
        node_assets_values={"cpu": 3},
    )

    assert infrastructure.nodes[SINGLETON_NODE]["cpu"] == 3


def test_singleton_strict_node():
    """A singleton node can be added strictly."""
    infrastructure = get_singleton(strict=True)

    assert list(infrastructure.nodes) == [SINGLETON_NODE]


def test_mirror_copies_the_federation_topology():
    """The mirror has the topology and assets of the federation, maximal."""
    federation = _federation("a", "b", "c")
    federation.add_connection("a", "b")

    infrastructure = get_mirror(federation)

    assert infrastructure.id == "mirror"
    assert set(infrastructure.nodes) == set(federation.nodes)
    assert set(infrastructure.edges) == set(federation.edges)
    assert set(infrastructure.node_assets) == set(federation.node_assets)
    assert set(infrastructure.edge_assets) == set(federation.edge_assets)
    assert infrastructure.attr_init == "max"


def test_mirror_assets_are_resources_not_requirements():
    """The mirror bounds are those of an infrastructure, not of a federation."""
    infrastructure = get_mirror(_federation("a"))
    reference = get_singleton()

    assert infrastructure.node_assets.upper_bound == reference.node_assets.upper_bound
    assert infrastructure.node_assets.upper_bound["availability"] == 1.0


def test_mirror_infrastructure_id_can_be_customized():
    """The identifier of the mirror infrastructure can be customized."""
    infrastructure = get_mirror(_federation("a"), infrastructure_id="copy", seed=1)

    assert infrastructure.id == "copy"


def test_mirror_accepts_custom_path_aggregators_and_strict():
    """The path aggregators and the strictness can be customized."""
    infrastructure = get_mirror(
        _federation("a"),
        strict=True,
        path_assets_aggregators=get_default_path_aggregators(),
    )

    assert list(infrastructure.nodes) == ["a"]
