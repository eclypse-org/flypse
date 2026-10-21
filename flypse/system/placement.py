"""Module for the placement strategies of FLYPSE.

The placement strategies map the entities of a federation onto the devices of
the infrastructure. They are ECLYPSE static strategies, whose mapping is
derived from the federation: every entity onto the single node of the
singleton infrastructure, or each entity onto its node of the mirror
infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eclypse.placement.strategies import StaticStrategy

from flypse.system.infrastructure import SINGLETON_NODE

if TYPE_CHECKING:
    from flypse.federation import Federation


def get_singleton_strategy(
    federation: Federation,
    node: str = SINGLETON_NODE,
) -> StaticStrategy:
    """Create the strategy placing every entity of a federation on one node.

    Args:
        federation (Federation): The federation whose entities are placed.
        node (str): The node hosting every entity. Defaults to the singleton
            node.

    Returns:
        StaticStrategy: The static mapping of every entity onto the node.
    """
    return StaticStrategy({entity_id: node for entity_id in federation.entities})


def get_mirror_strategy(federation: Federation) -> StaticStrategy:
    """Create the strategy placing every entity on its mirror node.

    The mirror infrastructure mirrors the federation, so each entity runs on
    the device named after it.

    Args:
        federation (Federation): The federation whose entities are placed.

    Returns:
        StaticStrategy: The identity mapping of each entity to its device.
    """
    return StaticStrategy({entity_id: entity_id for entity_id in federation.entities})
