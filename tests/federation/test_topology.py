"""Tests for the federation topology builders."""

from __future__ import annotations

import pytest

from flypse.federation import (
    Entity,
    get_ring,
    get_star,
    get_tree,
)


def _neighbors(federation, entity_id: str) -> set[str]:
    """Return the identifiers connected to an entity."""
    return set(federation.successors(entity_id))


def test_star_connects_the_center_to_each_leaf():
    """Leaves are connected to the center only."""
    federation = get_star(Entity("s"), [Entity("a"), Entity("b")])

    assert set(federation.entities) == {"s", "a", "b"}
    assert _neighbors(federation, "s") == {"a", "b"}
    assert _neighbors(federation, "a") == {"s"}


def test_builders_forward_the_federation_arguments():
    """Any other argument configures the federation."""
    federation = get_star(Entity("s"), [Entity("a")], federation_id="Star")

    assert federation.id == "Star"


def test_builders_reject_duplicate_identifiers():
    """Two entities with the same identifier are rejected."""
    with pytest.raises(ValueError, match="already in the federation"):
        get_star(Entity("s"), [Entity("a"), Entity("a")])


def test_tree_connects_each_entity_to_its_children():
    """A hierarchy of server, aggregators and clients."""
    federation = get_tree(
        Entity("server"),
        {
            "server": [Entity("g1"), Entity("g2")],
            "g1": [Entity("c1"), Entity("c2")],
            "g2": [Entity("c3")],
        },
    )

    assert _neighbors(federation, "server") == {"g1", "g2"}
    assert _neighbors(federation, "g1") == {"server", "c1", "c2"}
    assert _neighbors(federation, "c3") == {"g2"}


def test_tree_rejects_parents_outside_the_tree():
    """Children of an unknown parent cannot be connected."""
    with pytest.raises(ValueError, match="not in the tree"):
        get_tree(Entity("server"), {"other": [Entity("c1")]})


def test_tree_rejects_cycles():
    """An entity appearing twice would close a cycle."""
    with pytest.raises(ValueError, match="already in the federation"):
        get_tree(Entity("server"), {"server": [Entity("c1")], "c1": [Entity("server")]})


def test_ring_connects_each_entity_to_the_next():
    """Each peer is connected to its two neighbours in the ring."""
    federation = get_ring([Entity("a"), Entity("b"), Entity("c"), Entity("d")])

    assert _neighbors(federation, "a") == {"b", "d"}
    assert _neighbors(federation, "c") == {"b", "d"}


def test_ring_requires_three_entities():
    """Two entities cannot form a ring."""
    with pytest.raises(ValueError, match="at least three"):
        get_ring([Entity("a"), Entity("b")])
