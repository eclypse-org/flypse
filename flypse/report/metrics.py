"""Module for the federation metrics.

The federation metrics report what each entity observes of itself, read from
the live entity on its device after every step of the process: the number of
steps it completed, and its cumulative ``traffic`` (messages, bytes of model
and waiting times). The values only grow, so the difference between two
reports gives the activity in between.
"""

from __future__ import annotations

from dataclasses import fields
from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.report.metrics import metric

from flypse.federation.traffic import Traffic

if TYPE_CHECKING:
    from collections.abc import Callable

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
