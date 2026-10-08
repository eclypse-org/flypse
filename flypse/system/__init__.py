"""Abstractions of the system the federation runs on."""

from flypse.system.device import Device
from flypse.system.infrastructure import (
    SINGLETON_NODE,
    get_mirror,
    get_singleton,
)
from flypse.system.placement import (
    get_mirror_strategy,
    get_singleton_strategy,
)

__all__ = [
    "SINGLETON_NODE",
    "Device",
    "get_mirror",
    "get_mirror_strategy",
    "get_singleton",
    "get_singleton_strategy",
]
