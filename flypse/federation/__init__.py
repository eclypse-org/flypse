"""Package for the federation, its entities, their models and topologies."""

from flypse.federation.entity import Entity
from flypse.federation.federation import Federation
from flypse.federation.message import Message
from flypse.federation.model import (
    FederatedModel,
    ModelVersion,
)
from flypse.federation.topology import (
    get_ring,
    get_star,
    get_tree,
)

__all__ = [
    "Entity",
    "FederatedModel",
    "Federation",
    "Message",
    "ModelVersion",
    "get_ring",
    "get_star",
    "get_tree",
]
