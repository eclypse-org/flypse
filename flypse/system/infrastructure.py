"""Module for the infrastructure builders of FLYPSE.

The singleton infrastructure is the minimal system a federation can be
emulated on: a single node hosting every entity. It follows the signature of
the ECLYPSE infrastructure builders, so it can be swapped with any of them.
The mirror infrastructure is a copy of a federation, with one device per
entity and one link per connection.
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
    include_default_assets: bool = False,
    strict: bool = False,
    resource_init: InitPolicy = "min",
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Infrastructure:
    """Create an infrastructure with a single node hosting every entity.

    Args:
        infrastructure_id (str): The ID of the infrastructure. Defaults to
            "singleton".
        update_policies (UpdatePolicies): The graph update policies. Defaults
            to None.
        node_assets (dict[str, Asset] | None): The assets of the node.
            Defaults to None.
        link_assets (dict[str, Asset] | None): The assets of the links.
            Defaults to None.
        node_assets_values (dict[str, Any] | None): The values of the assets
            of the node. Defaults to None.
        include_default_assets (bool): Whether to include the default assets.
            Defaults to False.
        strict (bool): If True, raises an error if the node already exists.
            Defaults to False.
        resource_init (InitPolicy): The initialization policy for the
            resources. Defaults to "min".
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
    path_assets_aggregators: dict[str, Callable[[list[Any]], Any]] | None = None,
    path_algorithm: Callable[[nx.Graph, str, str], list[str]] | None = None,
    seed: int | None = None,
) -> Infrastructure:
    """Create an infrastructure mirroring the federation.

    The topology and the resources of the infrastructure are an exact copy of
    the topology and the requirements of the federation: every entity becomes
    a device with the same requirements as resources, and every connection
    becomes a link with the same requirements as resources.

    Args:
        federation (Federation): The federation to mirror.
        infrastructure_id (str): The ID of the infrastructure. Defaults to
            "mirror".
        update_policies (UpdatePolicies): The graph update policies. Defaults
            to None.
        strict (bool): If True, raises an error if a node or a link is not
            consistent. Defaults to False.
        path_assets_aggregators (dict[str, Callable[[list[Any]], Any]] | None):
            The aggregators of the path assets. Defaults to the ECLYPSE
            default ones (latency, bandwidth).
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
        node_assets=federation.node_assets,
        edge_assets=federation.edge_assets,
        path_assets_aggregators=(
            path_assets_aggregators or get_default_path_aggregators()
        ),
        path_algorithm=path_algorithm,
        seed=seed,
    )
    for node, assets in federation.nodes(data=True):
        infrastructure.add_node(node, strict=strict, **assets)
    for source, target, assets in federation.edges(data=True):
        infrastructure.add_edge(source, target, strict=strict, **assets)
    return infrastructure
