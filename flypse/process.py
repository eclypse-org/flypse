"""Module for the FederatedProcess class.

A federated process is the emulation of a federation: it binds together the
**system** (the infrastructure, whose nodes are devices, and the placement of
the entities onto them), the **federation** (the application, made of entities
and their interactions) and the **federation data**, and runs them through
ECLYPSE's remote emulation.

Being a :class:`~eclypse.simulation.Simulation`, it inherits the whole
ECLYPSE lifecycle (``register``, ``start``, ``step``, ``run``, reporting),
while the emulation of the federation is set up implicitly: the infrastructure
nodes become devices holding the federation data.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.remote.bootstrap import RemoteBootstrap
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)

from flypse.report import (
    get_evaluation_events,
    get_federation_metrics,
)
from flypse.seed import get_seed
from flypse.system import Device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eclypse.graph import Application
    from eclypse.placement.strategies import PlacementStrategy

    from flypse.data.dataset import FederatedDataset
    from flypse.federation import Federation
    from flypse.system import SystemConfig


class FederatedProcess(Simulation):
    """The emulation of a federation over a system of devices."""

    def __init__(
        self,
        federation: Federation,
        data: FederatedDataset,
        system: SystemConfig,
        include_federation_metrics: bool = True,
        eval_metrics: Sequence[str] = (),
        eval_every: int | None = None,
        **simulation_config: Any,
    ):
        """Create a federated process.

        The data is split across the entities of the federation holding data
        and given to every device, and the federation is registered, so the
        process is ready to ``run``.

        The process has a single seed, the ``seed`` of the simulation: every
        component given no seed of its own (the split of the data, the
        built-in infrastructures, the federation, the splitters and the
        ``rng`` of the entities) falls back to it.

        Args:
            federation (Federation): The application of the federation.
            data (FederatedDataset): The data of the federation.
            system (SystemConfig): The infrastructure hosting the federation,
                whose nodes are devices, and the placement of the entities
                onto them.
            include_federation_metrics (bool): Whether to report the
                federation metrics (steps and traffic of every entity),
                besides any given ``events``. Defaults to True.
            eval_metrics (Sequence[str]): The names of the metrics returned
                by the ``evaluate`` of the entities, each reported whenever
                the ``evaluate`` event is triggered (e.g.
                ``process.trigger("evaluate")``). Defaults to no metrics.
            eval_every (int | None): The number of steps of the process
                between two evaluations of the entities. Defaults to None,
                meaning the entities are evaluated only when the ``evaluate``
                event is triggered.
            **simulation_config: The arguments of the ECLYPSE
                ``SimulationConfig``, except ``remote``.
        """
        self.federation = federation
        self.data = data

        events = list(simulation_config.pop("events", None) or [])
        if include_federation_metrics:
            events += get_federation_metrics()
        events += get_evaluation_events(eval_metrics, eval_every)
        config = SimulationConfig(
            remote=RemoteBootstrap(node_class=Device, data=data),
            events=events,
            **simulation_config,
        )
        # expose the seed of the process before anything random is built
        config.prepare_runtime()

        holders = [eid for eid, e in federation.entities.items() if e.with_data]
        if holders:
            data._split(holders)
        if federation.seed is None:
            federation.rnd.seed(get_seed())
        infrastructure, self._placement_strategy = system.build(federation)
        super().__init__(infrastructure, config)
        self.register()

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
                Defaults to None, meaning the strategy of the system
                configuration.

        Raises:
            ValueError: If the application is not the federation of the
                process.
        """
        if application is not None and application is not self.federation:
            raise ValueError(
                "A FederatedProcess emulates a single federation: "
                f"cannot include '{application.id}' "
                f"while emulating '{self.federation.id}'."
            )
        super().register(
            self.federation,
            placement_strategy or self._placement_strategy,
        )
