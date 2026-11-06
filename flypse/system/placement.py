"""Module for the placement strategies of FLYPSE.

The placement strategies map the entities of a federation onto the devices of
the infrastructure. The singleton and mirror strategies are static, with a
mapping derived from the federation: every entity onto the single node of the
singleton infrastructure, or each entity onto its node of the mirror
infrastructure. The exclusive strategy places each entity on a device of its
own, on any infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from eclypse.placement.strategies import (
    PlacementStrategy,
    StaticStrategy,
)

from flypse.system.infrastructure import SINGLETON_NODE

if TYPE_CHECKING:
    from eclypse.graph import (
        Application,
        Infrastructure,
    )
    from eclypse.placement import (
        Placement,
        PlacementView,
    )

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


class ExclusiveStrategy(PlacementStrategy):
    """A strategy placing each entity on a device of its own.

    Every entity is placed on a distinct device satisfying its requirements.
    The assignment is a maximum matching between the entities and the devices
    able to host them, so it is found whenever one exists; otherwise, no
    entity is placed.
    """

    def place(
        self,
        infrastructure: Infrastructure,
        application: Application,
        _: dict[str, Placement],
        placement_view: PlacementView,
    ) -> dict[str, str]:
        """Place each entity of the federation on a device of its own.

        Args:
            infrastructure (Infrastructure): The infrastructure hosting the
                federation.
            application (Application): The federation to place.
            _ (dict[str, Placement]): The placements of the process.
            placement_view (PlacementView): The snapshot of the current state
                of the infrastructure.

        Returns:
            dict[str, str]: The device of each entity, or an empty mapping if
            the entities cannot all be placed on distinct devices.
        """
        # tagged nodes: an entity and a device may share an identifier
        entities = [("entity", entity) for entity in application.nodes]
        candidates = nx.Graph()
        candidates.add_nodes_from(entities)
        for entity, requirements in application.nodes(data=True):
            for device, resources in placement_view.residual.nodes(data=True):
                if infrastructure.node_assets.satisfies(resources, requirements):
                    candidates.add_edge(("entity", entity), ("device", device))
        matching = nx.bipartite.maximum_matching(candidates, top_nodes=entities)
        mapping = {
            entity: matching[("entity", entity)][1]
            for entity in application.nodes
            if ("entity", entity) in matching
        }
        return mapping if len(mapping) == len(entities) else {}
