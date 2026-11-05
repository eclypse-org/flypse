"""Tests for the FederatedSimulator class."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from typing import (
    Any,
    cast,
)
from unittest.mock import Mock

import pytest
from eclypse.report.backend import load_jsonl_rows
from eclypse.report.reporters import JSONReporter
from eclypse.simulation._simulator import RemoteSimulator
from eclypse.simulation._simulator.reporter import SimulationReporter

from flypse.system import (
    ENTITY_REPORT,
    FederatedSimulator,
)


@pytest.fixture
async def simulator(tmp_path) -> FederatedSimulator:
    """Return a simulator writing to a real reporter, with no Ray involved."""
    simulator = FederatedSimulator.__new__(FederatedSimulator)
    simulator._event_loop = asyncio.get_running_loop()
    simulator._reporter = SimulationReporter(tmp_path, {"json": JSONReporter})
    simulator._finished = Event()
    await simulator._reporter.start(simulator._event_loop)
    return simulator


def _sent(simulator: FederatedSimulator, **n_reports: int):
    """Set the entities the simulator got back, with the reports they sent."""
    entities = {
        entity_id: SimpleNamespace(n_reports=n) for entity_id, n in n_reports.items()
    }
    placement = SimpleNamespace(
        deployed=False, application=SimpleNamespace(services=entities)
    )
    simulator._manager = cast("Any", SimpleNamespace(placements={"Fed": placement}))


def _rows(tmp_path) -> list[dict]:
    """Return the rows of the service report."""
    return load_jsonl_rows(tmp_path / "json" / "service.jsonl", "service")


def test_federated_simulator_is_a_remote_simulator():
    """The simulator is the one of ECLYPSE, plus the reports of the entities."""
    assert issubclass(FederatedSimulator, RemoteSimulator)


async def test_report_lands_in_the_service_report(simulator, tmp_path):
    """A report is one row, identified by its entity and its number."""
    _sent(simulator, a=1)

    simulator.report("Fed", "a", 1, "2026-01-01T00:00:00", {"loss": 0.5})
    await simulator._finalize_shutdown()

    assert _rows(tmp_path) == [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_id": ENTITY_REPORT,
            "n_event": 1,
            "callback_id": ENTITY_REPORT,
            "application_id": "Fed",
            "service_id": "a",
            "value": {"loss": 0.5},
        }
    ]


async def test_shutdown_waits_for_the_reports_on_their_way(simulator, tmp_path):
    """The report closes only when all that the entities sent has arrived."""
    _sent(simulator, a=2, b=1)
    simulator.report("Fed", "a", 1, "t", {})
    loop = asyncio.get_running_loop()
    loop.call_later(0.01, simulator.report, "Fed", "b", 1, "t", {})
    loop.call_later(0.02, simulator.report, "Fed", "a", 2, "t", {})

    await simulator._finalize_shutdown()

    assert sorted((r["service_id"], r["n_event"]) for r in _rows(tmp_path)) == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
    ]


async def test_shutdown_waits_for_the_reports_even_after_a_crash(
    simulator, tmp_path, monkeypatch
):
    """A crashed entity fails the first cleanup, not the reports of the others."""
    crash = Mock(side_effect=[ValueError("crashed"), None, None])
    monkeypatch.setattr(simulator, "cleanup", crash)
    _sent(simulator, a=1)
    asyncio.get_running_loop().call_later(
        0.01, simulator.report, "Fed", "a", 1, "t", {}
    )

    await simulator._finalize_shutdown()

    assert len(_rows(tmp_path)) == 1


async def test_shutdown_without_reports_writes_nothing(simulator, tmp_path):
    """No report is needed for the simulator to stop."""
    _sent(simulator, a=0)

    await simulator._finalize_shutdown()

    assert not (tmp_path / "json" / "service.jsonl").exists()
