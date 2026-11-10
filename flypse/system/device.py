"""Module for the Device class.

A device is the node of the infrastructure an entity is deployed onto. In a
real federation the data lives on the device, not on the federated process:
every device is given the dataset of the federation, and each entity running
on it binds its own view of that dataset.

The entities deployed onto a device also share its slots, through its
``concurrency``: a device may admit only a given number of them at once in their
critical operations, such as a local training.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from eclypse.remote.node import RemoteNode

from flypse.system.concurrency import ConcurrencyHandler

if TYPE_CHECKING:
    from flypse.data.dataset import FederatedDataset


class Device(RemoteNode):
    """A node of the infrastructure holding the federation dataset."""

    def build(
        self,
        data: FederatedDataset | None = None,
        max_concurrency: int | None = None,
    ):
        """Store the federation dataset on the device, and set up its slots.

        Args:
            data (FederatedDataset | None): The dataset of the federation.
                Defaults to None.
            max_concurrency (int | None): The number of entities holding a
                slot at once. Defaults to None, meaning no limit.
        """
        super().build()
        self.data = data
        """The dataset of the federation."""
        self.concurrency = ConcurrencyHandler(max_concurrency)
        """The slots shared by the entities on the device."""
