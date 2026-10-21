"""Tests for the federation metrics."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from flypse.federation import Traffic
from flypse.report import get_federation_metrics

NAMES = {"steps", *(field.name for field in fields(Traffic))}
"""The names of the federation metrics."""


def test_metrics_are_remote_service_metrics():
    """One remote service metric per steps and traffic counter."""
    metrics = get_federation_metrics()

    assert {m.name for m in metrics} == NAMES
    assert all(m.is_metric and m.type == "service" and m._remote for m in metrics)


def test_metrics_are_created_anew_on_every_call():
    """Each process gets its own metric instances."""
    first = get_federation_metrics()
    second = get_federation_metrics()

    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_metrics_read_the_live_entity():
    """Each metric returns its quantity of the given entity."""
    entity = SimpleNamespace(
        step_count=4, traffic=Traffic(messages_sent=3, bytes_sent=96, send_time=0.5)
    )

    values = {m.name: m(entity) for m in get_federation_metrics()}

    assert values == {
        "steps": 4,
        "messages_sent": 3,
        "messages_received": 0,
        "bytes_sent": 96,
        "bytes_received": 0,
        "send_time": 0.5,
        "receive_time": 0.0,
    }
