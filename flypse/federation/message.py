"""Module for the Message class.

A message is what an entity receives from another entity of the federation:
a version of the sender's model, together with any data the sender attached
to it (e.g., the number of samples it was trained on). A control message
carries only data (e.g., a signal to stop), without a model.

A message is stamped with when it was sent, when it reached the device of the
receiving entity, and its size: e.g., how long a client took to answer, or the
bytes a federation exchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from flypse.federation.model import ModelVersion


@dataclass(frozen=True)
class Message:
    """A message received by an entity."""

    sender: str
    """The identifier of the sending entity."""

    record: ModelVersion | None
    """The version of the model sent, or None for a control message."""

    data: dict[str, Any]
    """The data attached by the sender."""

    sent_at: float
    """When the sender sent the message, as ``time.time()``."""

    received_at: float
    """When the message reached the device of the entity, as ``time.time()``.

    It is not when the entity received it: a message may wait for the entity to
    call ``receive``.
    """

    size: int
    """The size of the message, as serialized for the wire, in bytes.

    It is within a few hundred bytes of the size the communication cost of the
    message is computed from.
    """
