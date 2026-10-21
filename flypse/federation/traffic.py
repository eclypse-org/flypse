"""Module for the Traffic class.

The traffic of an entity is what it observes of its own communication: how
many messages it sent and received, how many bytes of model they carried, and
how long it waited for them. The counters only grow, so that they can be read
at any time (e.g., by the federation metrics) without resetting them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Traffic:
    """The cumulative communication counters of an entity."""

    messages_sent: int = 0
    """The number of messages sent, counted once per recipient."""

    messages_received: int = 0
    """The number of messages received."""

    bytes_sent: int = 0
    """The bytes of model sent, counted once per recipient."""

    bytes_received: int = 0
    """The bytes of model received."""

    send_time: float = 0.0
    """The seconds spent waiting for the messages sent to be delivered."""

    receive_time: float = 0.0
    """The seconds spent waiting for messages to arrive."""
