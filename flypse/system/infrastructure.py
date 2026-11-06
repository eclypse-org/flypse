"""Module for the infrastructure builders of FLYPSE.

The singleton infrastructure is the minimal system a federation can be
emulated on: a single node hosting every entity. The mirror infrastructure copies
the topology of a federation, with one device per entity and one link per
connection. Both have maximal resources, so that by default they can host any
federation, whose requirements are minimal.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.graph import Infrastructure
from eclypse.graph.assets.defaults import get_default_path_aggregators

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx
    from eclypse.graph.assets import Asset
    from eclypse.utils.types import (
        InitPolicy,
        UpdatePolicies,
    )

    from flypse.federation import Federation

SINGLETON_NODE = "singleton"
"""The identifier of the only node of the singleton infrastructure."""


def get_singleton(
    infrastructure_id: str = "singleton",
    update_policies: UpdatePolicies = None,
    node_assets: dict[str, Asset] | None = None,
    link_assets: dict[str, Asset] | None = None,
    node_assets_values: dict[str, Any] | None = None,
    include_default_assets: bool = True,
    strict: bool = False,
    resource_init: InitPolicy = "max",
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Infrastructure:
    """Create an infrastructure with a single node hosting every entity.

    Args:
        infrastructure_id (str): The identifier of the infrastructure. Defaults to
            "singleton".
        update_policies (UpdatePolicies): The functions changing the
            infrastructure, called at every process step. Defaults to None.
        node_assets (dict[str, Asset] | None): The resources of the node, on
            top of the default ones. Defaults to None.
        link_assets (dict[str, Asset] | None): The resources of the links, on
            top of the default ones. Defaults to None.
        node_assets_values (dict[str, Any] | None): The values of the resources
            of the node. Defaults to None.
        include_default_assets (bool): Whether to include the default
            resources. Defaults to True.
        strict (bool): If True, raises an error if the node already exists.
            Defaults to False.
        resource_init (InitPolicy): The initialisation policy for the
            resources. Defaults to "max".
        path_algorithm (Callable[[nx.Graph, str, str], list[str]] | None):
            The algorithm used to compute the paths. Defaults to None.
        seed (int | None): The seed for the random number generator. Defaults
            to None.

    Returns:
        Infrastructure: The singleton infrastructure.
    """
    infrastructure = Infrastructure(
        infrastructure_id=infrastructure_id,
        update_policies=update_policies,
        node_assets=node_assets,
        edge_assets=link_assets,
        include_default_assets=include_default_assets,
        resource_init=resource_init,
        path_algorithm=path_algorithm,
        seed=seed,
    )
    infrastructure.add_node(
        SINGLETON_NODE,
        strict=strict,
        **(node_assets_values or {}),
    )
    return infrastructure


def get_mirror(
    federation: Federation,
    infrastructure_id: str = "mirror",
    update_policies: UpdatePolicies = None,
    strict: bool = False,
    resource_init: InitPolicy = "max",
    path_assets_aggregators: dict[str, Callable[[list[Any]], Any]] | None = None,
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Infrastructure:
    """Create an infrastructure mirroring the topology of the federation.

    Every entity becomes a device with the same identifier, and every
    connection becomes a link. Devices and links have the same assets as the
    entities and the connections, initialised according to ``resource_init``.

    Args:
        federation (Federation): The federation to mirror.
        infrastructure_id (str): The identifier of the infrastructure. Defaults to
            "mirror".
        update_policies (UpdatePolicies): The functions changing the
            infrastructure, called at every process step. Defaults to None.
        strict (bool): If True, raises an error if a node or a link is not
            consistent. Defaults to False.
        resource_init (InitPolicy): The initialisation policy for the
            resources. Defaults to "max".
        path_assets_aggregators (dict[str, Callable[[list[Any]], Any]] | None):
            The functions combining the resources of the links along a path.
            Defaults to the ones of latency and bandwidth.
        path_algorithm (Callable[[nx.Graph, str, str], list[str]] | None):
            The algorithm used to compute the paths. Defaults to None.
        seed (int | None): The seed for the random number generator. Defaults
            to None.

    Returns:
        Infrastructure: The mirror infrastructure.
    """
    infrastructure = Infrastructure(
        infrastructure_id=infrastructure_id,
        update_policies=update_policies,
        # the federation holds its assets flipped, as requirements
        node_assets=federation.node_assets.flip(),
        edge_assets=federation.edge_assets.flip(),
        path_assets_aggregators=(
            path_assets_aggregators or get_default_path_aggregators()
        ),
        resource_init=resource_init,
        path_algorithm=path_algorithm,
        seed=seed,
    )
    for node in federation.nodes:
        infrastructure.add_node(node, strict=strict)
    for source, target in federation.edges:
        infrastructure.add_edge(source, target, strict=strict)
    return infrastructure
