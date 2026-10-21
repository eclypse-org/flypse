"""Module for the Message class.

A message is what an entity receives from another entity of the federation:
a version of the sender's model, together with any data the sender attached
to it (e.g., the number of samples it was trained on).
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

    record: ModelVersion
    """The version of the model sent."""

    data: dict[str, Any]
    """The data attached by the sender."""
