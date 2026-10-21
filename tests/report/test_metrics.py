"""Tests for the federation metrics."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from flypse.federation import Traffic
from flypse.report import (
    EVALUATE_EVENT,
    get_evaluation_events,
    get_federation_metrics,
)

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


class _Evaluating:
    """A live entity with a given evaluation."""

    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.eval_metrics = {"stale": 1.0}

    def evaluate(self):
        return self.evaluation


def test_evaluation_events_are_one_event_and_one_metric_per_name():
    """A manual remote evaluate event, then remote metrics cascaded from it."""
    evaluate, *metrics = get_evaluation_events(["loss", "accuracy"])

    assert evaluate.name == EVALUATE_EVENT
    assert evaluate.type == "service" and evaluate._remote
    assert not evaluate.is_post_event and not evaluate.trigger_bucket.triggers
    assert [m.name for m in metrics] == ["loss", "accuracy"]
    assert all(m.is_metric and m.type == "service" and m._remote for m in metrics)
    assert all(m._trigger(evaluate) for m in metrics)


def test_evaluate_event_keeps_the_evaluation_of_the_entity():
    """The evaluate event stores the results, read by the metrics."""
    evaluate, loss, accuracy = get_evaluation_events(["loss", "accuracy"])
    entity = _Evaluating({"loss": 0.5})

    evaluate(entity)

    assert entity.eval_metrics == {"loss": 0.5}
    assert loss(entity) == 0.5
    assert accuracy(entity) is None


def test_evaluate_event_clears_the_evaluation_of_a_non_evaluating_entity():
    """An entity returning None keeps no stale results."""
    evaluate, loss = get_evaluation_events(["loss"])
    entity = _Evaluating(None)

    evaluate(entity)

    assert entity.eval_metrics == {}
    assert loss(entity) is None
