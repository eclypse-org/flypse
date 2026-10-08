"""Tests for the Federation class."""

from __future__ import annotations

import pytest
from eclypse.graph import Application

from flypse.federation import Entity, Federation


def test_federation_is_an_application():
    """A federation is an ECLYPSE application."""
    assert issubclass(Federation, Application)


def test_new_federation_is_empty():
    """A federation starts without entities."""
    federation = Federation()

    assert federation.id == "Federation"
    assert list(federation.nodes) == []
    assert federation.entities == {}


def test_federation_id_is_forwarded():
    """The federation identifier is used as the application identifier."""
    assert Federation("MyFed").id == "MyFed"


def test_flows_are_forwarded():
    """The declared flows are installed on the federation."""
    federation = Federation(flows=[["a", "b"]])

    assert federation.flows == [["a", "b"]]


def test_add_entity_registers_it():
    """Adding an entity registers it as a node and as an entity."""
    federation = Federation("Fed")
    entity = Entity("a")

    federation.add_entity(entity)

    assert list(federation.nodes) == ["a"]
    assert federation.entities["a"] is entity
    assert federation.services["a"] is entity
    assert entity.application_id == "Fed"


def test_add_entity_with_assets():
    """The assets passed to add_entity become node requirements."""
    federation = Federation(include_default_assets=True)

    federation.add_entity(Entity("a"), cpu=1, ram=2.0)

    assert federation.nodes["a"]["cpu"] == 1
    assert federation.nodes["a"]["ram"] == 2.0


def test_add_entity_rejects_non_entity():
    """Only entities can be added to a federation."""
    federation = Federation()

    with pytest.raises(TypeError, match="Entity"):
        federation.add_entity("a")


def test_entity_cannot_join_two_federations():
    """An entity already assigned to a federation cannot be reassigned."""
    entity = Entity("a")
    Federation("First").add_entity(entity)

    with pytest.raises(ValueError, match="already assigned"):
        Federation("Second").add_entity(entity)


def test_add_connection_between_identifiers():
    """An interaction connects two entities and is symmetric by default."""
    federation = Federation(include_default_assets=True)
    federation.add_entity(Entity("a"))
    federation.add_entity(Entity("b"))

    federation.add_connection("a", "b", latency=10.0, bandwidth=5.0)

    assert federation.has_edge("a", "b")
    assert federation.has_edge("b", "a")
    assert federation.edges["a", "b"]["latency"] == 10.0
    assert federation.edges["b", "a"]["bandwidth"] == 5.0


def test_add_connection_accepts_entities():
    """Interactions can be declared using the entity objects."""
    federation = Federation()
    source, target = Entity("a"), Entity("b")
    federation.add_entity(source)
    federation.add_entity(target)

    federation.add_connection(source, target)

    assert federation.has_edge("a", "b")


def test_add_connection_can_be_directed():
    """A directed interaction is added in a single direction."""
    federation = Federation()
    federation.add_entity(Entity("a"))
    federation.add_entity(Entity("b"))

    federation.add_connection("a", "b", symmetric=False)

    assert federation.has_edge("a", "b")
    assert not federation.has_edge("b", "a")


def test_add_connection_requires_known_entities():
    """An interaction with an unknown entity is rejected."""
    federation = Federation()
    federation.add_entity(Entity("a"))

    with pytest.raises(ValueError, match="not found"):
        federation.add_connection("a", "ghost")


def test_has_entity_implementations():
    """Every node of the federation must have an entity implementation."""
    assert Federation().has_entity_implementations

    with_entities = Federation()
    with_entities.add_entity(Entity("a"))
    assert with_entities.has_entity_implementations

    incomplete = Federation()
    incomplete.add_node("ghost")
    assert not incomplete.has_entity_implementations


def test_default_assets_are_included():
    """Default assets are installed on the federation by default."""
    federation = Federation()

    federation.add_entity(Entity("a"))

    assert "cpu" in federation.nodes["a"]
    assert "availability" in federation.nodes["a"]
