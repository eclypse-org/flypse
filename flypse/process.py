"""Module for the FederatedProcess class.

A federated process is the emulation of a federation: it binds together the
**system** (the infrastructure, whose nodes are devices), the **federation**
(the application, made of entities and their interactions) and the
**federation data**, and runs them through ECLYPSE's remote emulation.

Being a :class:`~eclypse.simulation.Simulation`, it inherits the whole
ECLYPSE lifecycle (``register``, ``start``, ``step``, ``run``, reporting),
while the emulation of the federation is set up implicitly: the infrastructure
nodes become devices holding the federation data.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from eclypse.remote.bootstrap import RemoteBootstrap
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)

from flypse.system import (
    Device,
    get_mirror,
    get_mirror_strategy,
    get_singleton,
    get_singleton_strategy,
)

if TYPE_CHECKING:
    from eclypse.graph import (
        Application,
        Infrastructure,
    )
    from eclypse.placement.strategies import PlacementStrategy

    from flypse.data.dataset import FederatedDataset
    from flypse.federation import Federation


class FederatedProcess(Simulation):
    """The emulation of a federation over a system of devices."""

    def __init__(
        self,
        federation: Federation,
        data: FederatedDataset,
        infrastructure: Infrastructure | Literal["singleton", "mirror"],
        **simulation_config: Any,
    ):
        """Create a federated process.

        The data is split across the entities of the federation and given to
        every device, and the federation is registered, so the process is
        ready to ``run``.

        Args:
            federation (Federation): The application of the federation.
            data (FederatedDataset): The data of the federation.
            infrastructure (Infrastructure | Literal["singleton", "mirror"]):
                The infrastructure hosting the federation; each of its nodes
                is a device. With "singleton", a single-node infrastructure is
                created and all entities are placed on that node. With
                "mirror", a copy of the federation topology and requirements
                is created and each entity is placed on its mirror node. With
                an ``Infrastructure``, a ``default_strategy`` must be given in
                the simulation configuration.
            **simulation_config: The arguments of the ECLYPSE
                ``SimulationConfig``, except ``remote``.
        """
        self._federation = federation
        self._data = data

        if federation.entities:
            data._split(list(federation.entities))

        if isinstance(infrastructure, str):
            if infrastructure == "mirror":
                infrastructure = get_mirror(federation)
                self._placement_strategy: PlacementStrategy | None = (
                    get_mirror_strategy(federation) if federation.entities else None
                )
            else:
                infrastructure = get_singleton()
                self._placement_strategy = (
                    get_singleton_strategy(federation) if federation.entities else None
                )
        else:
            self._placement_strategy = None

        config = SimulationConfig(
            remote=RemoteBootstrap(node_class=Device, data=data),
            **simulation_config,
        )
        super().__init__(infrastructure, config)
        self.register()

    @property
    def federation(self) -> Federation:
        """Get the federation emulated by the process.

        Returns:
            Federation: The federation of the process.
        """
        return self._federation

    @property
    def data(self) -> FederatedDataset:
        """Get the data of the federation.

        Returns:
            FederatedDataset: The data given to the devices.
        """
        return self._data

    def register(
        self,
        application: Application | None = None,
        placement_strategy: PlacementStrategy | None = None,
    ):
        """Register the federation of the process for emulation.

        The federation is the one given at construction, so it can be
        omitted. A federated process emulates a single federation: any other
        application is rejected.

        Args:
            application (Application | None): The federation to register. It
                must be the federation of the process. Defaults to None,
                meaning the federation of the process.
            placement_strategy (PlacementStrategy | None): The strategy used
                to place the entities of the federation on the devices.
                Defaults to None, meaning the strategy of the "singleton" or
                "mirror" infrastructure, if one was given at construction.

        Raises:
            ValueError: If the application is not the federation of the
                process.
        """
        if application is not None and application is not self._federation:
            raise ValueError(
                "A FederatedProcess emulates a single federation: "
                f"cannot include '{application.id}' "
                f"while emulating '{self._federation.id}'."
            )
        super().register(
            self._federation,
            placement_strategy or self._placement_strategy,
        )
