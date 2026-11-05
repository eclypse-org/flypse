"""Package for the devices, infrastructures, placement and simulator."""

from flypse.system.config import SystemConfig
from flypse.system.device import Device
from flypse.system.infrastructure import (
    SINGLETON_NODE,
    get_mirror,
    get_singleton,
)
from flypse.system.placement import (
    ExclusiveStrategy,
    get_mirror_strategy,
    get_singleton_strategy,
)
from flypse.system.simulator import (
    ENTITY_REPORT,
    FederatedSimulator,
)

__all__ = [
    "ENTITY_REPORT",
    "SINGLETON_NODE",
    "Device",
    "ExclusiveStrategy",
    "FederatedSimulator",
    "SystemConfig",
    "get_mirror",
    "get_mirror_strategy",
    "get_singleton",
    "get_singleton_strategy",
]
