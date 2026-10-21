"""Tests for the SystemConfig class."""

from __future__ import annotations

import pytest
from eclypse.graph import Infrastructure
from eclypse.placement.strategies import (
    RandomStrategy,
    StaticStrategy,
)

from flypse.federation import Entity, Federation
from flypse.system import (
    SINGLETON_NODE,
    SystemConfig,
)


def _federation(*entity_ids: str) -> Federation:
    """Return a federation with one entity per given identifier."""
    federation = Federation("Fed")
    for entity_id in entity_ids:
        federation.add_entity(Entity(entity_id))
    return federation


def test_default_is_the_mirror():
    """Without arguments, the federation runs on its mirror."""
    infrastructure, strategy = SystemConfig().build(_federation("a", "b"))

    assert infrastructure.id == "mirror"
    assert isinstance(strategy, StaticStrategy)
    assert strategy.mapping == {"a": "a", "b": "b"}


def test_singleton_infers_its_strategy():
    """The singleton strategy is inferred for the singleton infrastructure."""
    infrastructure, strategy = SystemConfig("singleton").build(_federation("a"))

    assert list(infrastructure.nodes) == [SINGLETON_NODE]
    assert isinstance(strategy, StaticStrategy)
    assert strategy.mapping == {"a": SINGLETON_NODE}


def test_an_explicit_strategy_wins():
    """A given strategy is used even with a built-in infrastructure."""
    given = RandomStrategy(seed=1)

    _, strategy = SystemConfig("mirror", given).build(_federation("a"))

    assert strategy is given


def test_no_strategy_without_entities():
    """A federation without entities has nothing to place."""
    _, strategy = SystemConfig("singleton").build(Federation("Fed"))

    assert strategy is None


def test_custom_infrastructure_is_used_as_is():
    """A custom infrastructure and its strategy are returned unchanged."""
    custom = Infrastructure("Infra")
    given = RandomStrategy(seed=1)

    assert SystemConfig(custom, given).build(_federation("a")) == (custom, given)


def test_custom_infrastructure_requires_a_strategy():
    """A custom infrastructure without a strategy is rejected."""
    with pytest.raises(ValueError, match="placement strategy is required"):
        SystemConfig(Infrastructure("Infra"))


def test_unknown_infrastructure_name_is_rejected():
    """Only the built-in infrastructures can be given by name."""
    with pytest.raises(ValueError, match="Unknown infrastructure"):
        SystemConfig("ring")  # type: ignore[arg-type]
