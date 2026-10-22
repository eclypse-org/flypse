"""Module for the Federation class.

A federation is an ECLYPSE application whose nodes are federated entities and
whose edges are the interactions among them. This class verticalizes the
``Application`` API around the federated setting: entities are added with
``add_entity`` and their interactions with ``add_connection``, so that a
federation can be built, and then emulated by ECLYPSE, without dealing with
the generic service and application vocabulary.

The federation is only the *application* side of the emulation: it knows the
entities and how they interact, not where they run nor the data they will
find there (that is the infrastructure side, see ``flypse.system``).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

from eclypse.graph import Application

from flypse.federation.entity import Entity

if TYPE_CHECKING:
    from eclypse.graph.assets import Asset
    from eclypse.utils.types import (
        InitPolicy,
        UpdatePolicies,
    )


class Federation(Application):
    """A federated learning application: entities and their interactions."""

    def __init__(
        self,
        federation_id: str = "Federation",
        update_policies: UpdatePolicies = None,
        node_assets: dict[str, Asset] | None = None,
        edge_assets: dict[str, Asset] | None = None,
        include_default_assets: bool = True,
        requirement_init: InitPolicy = "min",
        flows: list[list[str]] | None = None,
        seed: int | None = None,
    ):
        """Create a new federation.

        Args:
            federation_id (str): The identifier of the federation. Defaults
                to "Federation".
            update_policies (UpdatePolicies): The graph update policies
                executed during ``evolve()``. Defaults to None.
            node_assets (dict[str, Asset] | None): The assets of the
                entities. Defaults to None.
            edge_assets (dict[str, Asset] | None): The assets of the
                interactions. Defaults to None.
            include_default_assets (bool): Whether to include the default
                assets. Defaults to True.
            requirement_init (InitPolicy): The initialization of the
                requirements. Defaults to "min".
            flows (list[list[str]] | None): The flows of the federation.
                Defaults to None.
            seed (int | None): The seed for the random number generator.
                Defaults to None.
        """
        super().__init__(
            application_id=federation_id,
            update_policies=update_policies,
            node_assets=node_assets,
            edge_assets=edge_assets,
            include_default_assets=include_default_assets,
            requirement_init=requirement_init,
            flows=flows,
            seed=seed,
        )

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities of the federation, keyed by identifier.

        Returns:
            dict[str, Entity]: The entities of the federation.
        """
        return cast("dict[str, Entity]", self.services)

    def add_entity(self, entity: Entity, **assets):
        """Add an entity to the federation.

        Args:
            entity (Entity): The entity to add.
            **assets: The assets required by the entity.

        Raises:
            TypeError: If the entity is not an :class:`Entity`.
            ValueError: If an entity with the same identifier is already in
                the federation.
        """
        if not isinstance(entity, Entity):
            raise TypeError(f"Expected Entity instance, got {type(entity).__name__}")
        if entity.id in self.entities:
            raise ValueError(f"Entity {entity.id!r} is already in the federation.")
        super().add_service(entity, **assets)

    def add_connection(
        self,
        source: Entity | str,
        target: Entity | str,
        symmetric: bool = True,
        **assets,
    ):
        """Add an interaction between two entities of the federation.

        Args:
            source (Entity | str): The source entity, or its identifier.
            target (Entity | str): The target entity, or its identifier.
            symmetric (bool): Whether the interaction is bidirectional.
                Defaults to True.
            **assets: The assets of the interaction.
        """
        super().add_edge(
            _entity_id(source),
            _entity_id(target),
            symmetric=symmetric,
            **assets,
        )

    @property
    def has_entity_implementations(self) -> bool:
        """Whether every entity of the federation has an implementation.

        This is required for remote emulation.

        Returns:
            bool: Whether every entity of the federation has an
            implementation.
        """
        return self.has_service_implementations


def _entity_id(entity: Entity | str) -> str:
    """Get the identifier of an entity, given the entity or its identifier.

    Args:
        entity (Entity | str): The entity, or its identifier.

    Returns:
        str: The identifier of the entity.
    """
    return entity.id if isinstance(entity, Entity) else entity
