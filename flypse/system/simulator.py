"""Module for the FederatedSimulator class.

The federated simulator is the part of a federated process that runs the
emulation, with an endpoint the entities push their reports to. A report does not
fit the columns of the tabular formats, so it is handed to the JSON reporter of
the process, and it lands in the ``service`` report as a single row, whatever the
``report_format`` of the process.

A report needs no event: it is written even if it arrives while the process is
stopping. The simulator counts them, so that it closes the report only once it
has received everything the stopped entities sent.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from eclypse.simulation._simulator import RemoteSimulator
from eclypse.utils.constants import FLOAT_EPSILON
from eclypse.utils.defaults import JSON_REPORT_DIR

if TYPE_CHECKING:
    from flypse.federation import Entity

ENTITY_REPORT = "entity_report"
"""The name of the reports of the entities in the ``service`` report."""


class FederatedSimulator(RemoteSimulator):
    """A remote simulator receiving the reports of the entities."""

    _n_reports = 0

    def report(
        self,
        application_id: str,
        entity_id: str,
        n_report: int,
        timestamp: str,
        values: dict[str, Any],
    ):
        """Hand a report of an entity to the JSON reporter of the process.

        Args:
            application_id (str): The identifier of the federation.
            entity_id (str): The identifier of the entity.
            n_report (int): The progressive number of the report for the entity.
            timestamp (str): When the entity made the report.
            values (dict[str, Any]): The reported values.
        """
        row = {
            "timestamp": timestamp,
            "event_name": ENTITY_REPORT,
            "event_idx": n_report,
            "callback_name": ENTITY_REPORT,
            "data": [application_id, entity_id, values],
        }
        # this runs on the loop of the actor, the reporter on the one of the simulation
        self._event_loop.call_soon_threadsafe(self._enqueue, row)

    def _enqueue(self, row: dict[str, Any]):
        """Queue a row for the JSON reporter.

        Args:
            row (dict[str, Any]): The row of the ``service`` report.
        """
        self._n_reports += 1
        self._reporter._ensure_queue(JSON_REPORT_DIR, "service").put_nowait([row])

    async def _finalize_shutdown(self):
        """Close the report once all the reports of the entities have arrived."""
        try:
            # eclypse#52: a crashed entity fails the first stop, before any undeploy
            with suppress(ValueError):
                self.cleanup()
            self.cleanup()
            # the stopped entities came back with the number of reports they sent
            sent = sum(
                cast("Entity", entity).n_reports
                for placement in self.placements.values()
                for entity in placement.application.services.values()
            )
            while self._n_reports < sent:
                await asyncio.sleep(FLOAT_EPSILON)
        finally:
            await super()._finalize_shutdown()
