"""Module for the federation topology builders.

The builders create a federation from its entities, connecting them according
to a common topology: a star (e.g., a server and its clients), a tree (e.g., a
server, group aggregators and their clients), or a ring (e.g., peers of a
decentralised federation). Any other argument is passed to the
:class:`~flypse.federation.federation.Federation`.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from flypse.federation.federation import Federation

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

    from flypse.federation.entity import Entity

MIN_RING_SIZE = 3
"""The minimum number of entities of a ring."""


def get_star(
    center: Entity, leaves: Sequence[Entity], **federation_args: Any
) -> Federation:
    """Create a federation connecting a centre to each of the leaves.

    Args:
        center (Entity): The entity at the centre (e.g., the server).
        leaves (Sequence[Entity]): The entities connected to the centre
            (e.g., the clients).
        **federation_args: The arguments of the federation.

    Returns:
        Federation: The star federation.

    Raises:
        ValueError: If two entities share an identifier.
    """
    federation = _federation([center, *leaves], federation_args)
    for leaf in leaves:
        federation.add_connection(center, leaf)
    return federation


def get_tree(
    root: Entity,
    children: Mapping[str, Sequence[Entity]],
    **federation_args: Any,
) -> Federation:
    """Create a federation connecting each entity to its children.

    Args:
        root (Entity): The entity at the root (e.g., the server).
        children (Mapping[str, Sequence[Entity]]): The children of each
            entity, keyed by the identifier of their parent.
        **federation_args: The arguments of the federation.

    Returns:
        Federation: The tree federation.

    Raises:
        ValueError: If two entities share an identifier, or a parent is not
            in the tree.

    Example:
        .. code-block:: python

            federation = get_tree(
                Server("server"),
                {
                    "server": [Aggregator("edge_0"), Aggregator("edge_1")],
                    "edge_0": [
                        Client("client_0", "edge_0"),
                        Client("client_1", "edge_0"),
                    ],
                    "edge_1": [
                        Client("client_2", "edge_1"),
                        Client("client_3", "edge_1"),
                    ],
                },
            )
    """
    federation = _federation(
        [root, *(child for kids in children.values() for child in kids)],
        federation_args,
    )
    reached = [root.id]
    for parent in reached:  # grows while walking, level by level
        for child in children.get(parent, ()):
            federation.add_connection(parent, child)
            reached.append(child.id)
    unreachable = sorted(set(children) - set(reached))
    if unreachable:
        raise ValueError(f"Parents not in the tree: {unreachable}.")
    return federation


def get_ring(entities: Sequence[Entity], **federation_args: Any) -> Federation:
    """Create a federation connecting each entity to the next, in a ring.

    Args:
        entities (Sequence[Entity]): The entities, in ring order.
        **federation_args: The arguments of the federation.

    Returns:
        Federation: The ring federation.

    Raises:
        ValueError: If there are fewer than three entities, or two entities
            share an identifier.
    """
    if len(entities) < MIN_RING_SIZE:
        raise ValueError("A ring requires at least three entities.")
    federation = _federation(entities, federation_args)
    for entity, following in zip(entities, [*entities[1:], entities[0]], strict=True):
        federation.add_connection(entity, following)
    return federation


def _federation(entities: Sequence[Entity], args: dict[str, Any]) -> Federation:
    """Create a federation including the given entities.

    Args:
        entities (Sequence[Entity]): The entities of the federation.
        args (dict[str, Any]): The arguments of the federation.

    Returns:
        Federation: The federation, with its entities and no connections.

    Raises:
        ValueError: If two entities share an identifier.
    """
    federation = Federation(**args)
    for entity in entities:
        federation.add_entity(entity)
    return federation
