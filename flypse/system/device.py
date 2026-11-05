"""Module for the Device class.

A device is the node of the infrastructure an entity is deployed onto. In a
real federation the data lives on the device, not on the federated process:
every device is given the dataset of the federation, and each entity running
on it binds its own view of that dataset.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from eclypse.remote.node import RemoteNode

if TYPE_CHECKING:
    from flypse.data.dataset import FederatedDataset


class Device(RemoteNode):
    """A node of the infrastructure holding the federation dataset."""

    def build(self, data: FederatedDataset | None = None):
        """Store the federation dataset on the device.

        Args:
            data (FederatedDataset | None): The dataset of the federation.
                Defaults to None.
        """
        super().build()
        self.data = data
        """The dataset of the federation."""
