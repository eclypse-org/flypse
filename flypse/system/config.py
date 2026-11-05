"""Module for the SystemConfig class.

The system configuration describes where a federation runs: the infrastructure
hosting the entities, and the placement strategy mapping the entities onto its
devices. It keeps the two consistent: the "singleton" and "mirror"
infrastructures come with their own placement strategy, which is inferred when
none is given, while any other infrastructure requires an explicit one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
)

from flypse.seed import get_seed
from flypse.system.infrastructure import (
    get_mirror,
    get_singleton,
)
from flypse.system.placement import (
    get_mirror_strategy,
    get_singleton_strategy,
)

if TYPE_CHECKING:
    from eclypse.graph import Infrastructure
    from eclypse.placement.strategies import PlacementStrategy

    from flypse.federation import Federation


@dataclass
class SystemConfig:
    """The configuration of the system a federation runs on.

    Raises:
        ValueError: If the infrastructure is an unknown name, or an
            ``Infrastructure`` given without a placement strategy.

    Example:
        .. code-block:: python

            SystemConfig("singleton")
            SystemConfig(infrastructure, ExclusiveStrategy())
    """

    infrastructure: Infrastructure | Literal["singleton", "mirror"] = "mirror"
    """The infrastructure hosting the federation; each of its nodes is a device.

    With "singleton", a single-node infrastructure hosts every entity. With
    "mirror", a copy of the federation topology hosts each entity on its own
    device. Defaults to "mirror".
    """

    placement_strategy: PlacementStrategy | None = None
    """The strategy placing the entities onto the devices.

    Defaults to None, meaning the strategy of the "singleton" or "mirror"
    infrastructure.
    """

    def __post_init__(self):
        """Check the consistency of the infrastructure and the strategy."""
        if isinstance(self.infrastructure, str):
            if self.infrastructure not in ("singleton", "mirror"):
                raise ValueError(
                    f"Unknown infrastructure '{self.infrastructure}': "
                    "expected 'singleton' or 'mirror'."
                )
        elif self.placement_strategy is None:
            raise ValueError(
                "A placement strategy is required for a custom infrastructure."
            )

    def build(
        self, federation: Federation
    ) -> tuple[Infrastructure, PlacementStrategy | None]:
        """Build the infrastructure and the placement strategy of a federation.

        The "singleton" and "mirror" infrastructures are seeded with the seed
        of the process.

        Args:
            federation (Federation): The federation to host.

        Returns:
            tuple[Infrastructure, PlacementStrategy | None]: The
            infrastructure, and the strategy placing the entities onto it
            (None when the federation has no entities to place).
        """
        strategy = self.placement_strategy
        if not isinstance(self.infrastructure, str):
            return self.infrastructure, strategy
        infer = strategy is None and bool(federation.entities)
        if self.infrastructure == "singleton":
            infrastructure = get_singleton(seed=get_seed())
            if infer:
                strategy = get_singleton_strategy(federation)
        else:
            infrastructure = get_mirror(federation, seed=get_seed())
            if infer:
                strategy = get_mirror_strategy(federation)
        return infrastructure, strategy
