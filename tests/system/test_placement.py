"""Tests for the placement strategy builders."""

from __future__ import annotations

import pytest

from flypse.federation import Entity, Federation
from flypse.system import (
    SINGLETON_NODE,
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
