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

from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.remote.bootstrap import RemoteBootstrap
from eclypse.report.backend import load_jsonl_rows
from eclypse.report.reporters import JSONReporter
from eclypse.simulation import (
    Simulation,
    SimulationConfig,
)
from eclypse.utils.defaults import JSON_REPORT_DIR

from flypse.seed import get_seed
from flypse.system import (
    ENTITY_REPORT,
    Device,
    FederatedSimulator,
    SystemConfig,
)

if TYPE_CHECKING:
    from eclypse.graph import Application
    from eclypse.placement.strategies import PlacementStrategy

    from flypse.data.dataset import FederatedDataset
    from flypse.federation import Federation


class FederatedProcess(Simulation):
    """The emulation of a federation over a system of devices."""

    def __init__(
        self,
        federation: Federation,
        data: FederatedDataset,
        system: SystemConfig | None = None,
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
            system (SystemConfig | None): The infrastructure hosting the
                federation, whose nodes are devices, and the placement of the
                entities onto them. Defaults to None, meaning the mirror
                infrastructure (``SystemConfig()``).
            **simulation_config: The arguments of the ECLYPSE
                ``SimulationConfig``, except ``remote``.
        """
        self.federation = federation
        self.data = data

        reporters = simulation_config.pop("reporters", None) or {}
        config = SimulationConfig(
            remote=RemoteBootstrap(
                sim_class=FederatedSimulator,
                node_class=Device,
                data=data,
            ),
            reporters={JSON_REPORT_DIR: JSONReporter, **reporters},
            **simulation_config,
        )
        # expose the seed of the process before anything random is built
        config.prepare_runtime()

        holders = [eid for eid, e in federation.entities.items() if e.with_data]
        if holders:
            data._split(holders)
        if federation.seed is None:
            federation.rnd.seed(get_seed())
        system = system or SystemConfig()
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

    @property
    def reported(self) -> list[dict[str, Any]]:
        """The values reported by the entities of the finished process.

        They are kept apart from ``report``, in the JSON report, as each one is
        a single record: the values of a call of ``Entity.report``, with its
        ``entity_id``, ``node_id``, ``timestamp`` and ``n_report``, which
        identifies it among those of the entity.

        Returns:
            list[dict[str, Any]]: One record per report, oldest first.
        """
        self.wait()
        path = self.path / JSON_REPORT_DIR / "service.jsonl"
        rows = load_jsonl_rows(path, "service") if path.exists() else []
        return sorted(
            (
                {
                    **row["value"],
                    "entity_id": row["service_id"],
                    "timestamp": row["timestamp"],
                    "n_report": row["n_event"],
                }
                for row in rows
                if row["callback_id"] == ENTITY_REPORT
            ),
            key=itemgetter("timestamp"),
        )
