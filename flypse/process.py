"""Module for the FederatedProcess class.

A federated process is the emulation of a federation: it binds together the
**federation** (the entities and their connections), the **federation data** and
the **system** (the infrastructure, whose nodes are devices, and the placement of
the entities onto them), and runs them. The emulation is set up implicitly: the
nodes of the infrastructure become devices holding the federation data.
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
    """The emulation of a federation over a system of devices.

    Example:
        .. code-block:: python

            process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
            process.run(seconds=10)
    """

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

        The process has a single seed, its ``seed`` option: every
        component given no seed of its own (the split of the data, the
        built-in infrastructures, the federation, the splitters and the
        ``rng`` of the entities) falls back to it.

        Args:
            federation (Federation): The federation to emulate.
            data (FederatedDataset): The data of the federation.
            system (SystemConfig | None): The infrastructure hosting the
                federation, whose nodes are devices, the placement of the
                entities onto them, and the slots of the devices.
                Defaults to None, meaning the mirror infrastructure with no
                limit (``SystemConfig()``).
            **simulation_config: The options of the run, such as ``seed`` or
                ``include_default_metrics``. ``step_every_ms`` must be given to
                run for a given time: without it, entities are not deployed.
        """
        self.federation = federation
        """The federation of the process."""
        self.data = data
        """The data of the federation."""

        system = system or SystemConfig()
        reporters = simulation_config.pop("reporters", None) or {}
        config = SimulationConfig(
            remote=RemoteBootstrap(
                sim_class=FederatedSimulator,
                node_class=Device,
                data=data,
                max_concurrency=system.max_concurrency,
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
        one is rejected.

        Args:
            application (Application | None): The federation to register. It
                must be the federation of the process. Defaults to None,
                meaning the federation of the process.
            placement_strategy (PlacementStrategy | None): The strategy used
                to place the entities of the federation on the devices.
                Defaults to None, meaning the strategy of the system
                configuration.

        Raises:
            ValueError: If it is not the federation of the process.
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

        Example:
            .. code-block:: python

                for record in process.reported:
                    print(record["entity_id"], record["round"], record["loss"])
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
