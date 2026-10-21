"""Module for the federation metrics.

The federation metrics report what each entity observes of itself, read from
the live entity on its device after every step of the process: the number of
steps it completed, and its cumulative ``traffic`` (messages, bytes of model
and waiting times). The values only grow, so the difference between two
reports gives the activity in between.

The evaluation events report what each entity measures of itself when
evaluated: the ``evaluate`` event calls ``evaluate`` on the live entities,
which keep the results in their ``eval_metrics``, and one metric per name,
fired right after it, reads them.
"""

from __future__ import annotations

from dataclasses import fields
from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.report.metrics import metric
from eclypse.workflow.event.wrapper import EventWrapper

from flypse.federation.traffic import Traffic

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from eclypse.workflow.event import EclypseEvent

    from flypse.federation import Entity


def get_federation_metrics() -> list[EclypseEvent]:
    """Create the federation metrics, one per reported quantity.

    The metrics are created anew on every call, so that each process gets its
    own instances.

    Returns:
        list[EclypseEvent]: The ``steps`` metric, and one metric per counter
        of the traffic of the entities, named after it.
    """
    return [
        metric.service(_steps, name="steps", remote=True),
        *(
            metric.service(_traffic_counter(field.name), name=field.name, remote=True)
            for field in fields(Traffic)
        ),
    ]


EVALUATE_EVENT = "evaluate"
"""The name of the event evaluating the entities."""


def get_evaluation_events(names: Sequence[str]) -> list[EclypseEvent]:
    """Create the evaluation event, and one metric per evaluation metric.

    The ``evaluate`` event has no trigger of its own: it fires when the
    process triggers it. The events are created anew on every call, so that
    each process gets its own instances.

    Args:
        names (Sequence[str]): The names of the evaluation metrics returned
            by ``evaluate``.

    Returns:
        list[EclypseEvent]: The ``evaluate`` event, followed by one metric
        per name, reading the latest evaluation of the entities.
    """
    return [
        EventWrapper(
            _evaluate,
            name=EVALUATE_EVENT,
            triggers=[],
            event_type="service",
            remote=True,
        ),
        *(
            metric.service(
                _eval_metric(name),
                name=name,
                activates_on=EVALUATE_EVENT,
                remote=True,
            )
            for name in names
        ),
    ]


def _evaluate(entity: Entity, **_: Any):
    """Evaluate the entity, keeping the results in its ``eval_metrics``.

    Args:
        entity (Entity): The live entity.
        **_: The data of the triggering event, unused.
    """
    entity.eval_metrics = entity.evaluate() or {}


def _eval_metric(name: str) -> Callable[..., Any]:
    """Create the reader of an evaluation metric of an entity.

    Args:
        name (str): The name of the evaluation metric.

    Returns:
        Callable[..., Any]: The function returning the metric of an entity,
        or None (no row) if its latest evaluation lacks it.
    """

    def read(entity: Entity, **_: Any) -> Any:
        return entity.eval_metrics.get(name)

    return read


def _steps(entity: Entity, **_: Any) -> int:
    """Return the number of steps completed by the entity.

    Args:
        entity (Entity): The live entity.
        **_: The data of the triggering event, unused.

    Returns:
        int: The number of steps.
    """
    return entity.step_count


def _traffic_counter(name: str) -> Callable[..., Any]:
    """Create the reader of a counter of the traffic of an entity.

    Args:
        name (str): The name of the counter.

    Returns:
        Callable[..., Any]: The function returning the counter of an entity.
    """

    def read(entity: Entity, **_: Any) -> Any:
        return getattr(entity.traffic, name)

    return read
