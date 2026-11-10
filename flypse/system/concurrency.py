"""Module for the ConcurrencyHandler class.

The concurrency handler of a device limits how many of the entities deployed onto
it are in a critical operation at once, such as a local training: each one holds
one of the ``max_concurrency`` slots of the device within an ``async with``
block. Slots are given in the order they are asked, straight to the entity
waiting the longest, so that no entity waits forever.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections import deque


class ConcurrencyHandler:
    """The slots of a device, shared by the entities on it.

    Example:
        .. code-block:: python

            async with self.concurrency:
                train(self.model.model, self.data)
    """

    def __init__(self, max_concurrency: int | None = None):
        """Initialise the concurrency handler of a device.

        Args:
            max_concurrency (int | None): The number of entities holding a
                slot at once. Defaults to None, meaning no limit.

        Raises:
            ValueError: If ``max_concurrency`` is below 1.
        """
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be at least 1, got {max_concurrency}."
            )
        self.max_concurrency = max_concurrency
        """The number of entities holding a slot at once."""
        self.holders: set[str | None] = set()
        """The entities holding a slot."""
        self.entity_id: str | None = None
        """The entity using the handler, once bound."""
        self._queue: deque[tuple[str | None, asyncio.Future]] = deque()
        # each entity runs on its own thread and event loop
        self._lock = threading.Lock()

    def bind(self, entity_id: str) -> ConcurrencyHandler:
        """Return the handler as used by an entity.

        The returned handler shares the slots of this one: only the entity on
        whose behalf it holds and releases them differs.

        Args:
            entity_id (str): The entity using the handler.

        Returns:
            ConcurrencyHandler: The handler used by the entity.
        """
        bound = copy.copy(self)  # shares the state, which is only changed in place
        bound.entity_id = entity_id
        return bound

    async def __aenter__(self):
        """Wait for a slot."""
        await self.acquire()

    async def __aexit__(self, *_exc_info: object):
        """Give back the slot."""
        self.release()

    async def acquire(self):
        """Wait for a slot, after the entities asking before.

        Raises:
            RuntimeError: If the entity holds, or waits for, a slot already.
        """
        with self._lock:
            if self.entity_id in self.holders or any(
                waiting == self.entity_id for waiting, _ in self._queue
            ):
                raise RuntimeError(
                    f"Entity {self.entity_id} already holds or waits for a slot."
                )
            # a released slot goes to the queue first: a free one means no queue
            if self.max_concurrency is None or len(self.holders) < self.max_concurrency:
                self.holders.add(self.entity_id)
                return
            waiter = asyncio.get_running_loop().create_future()
            self._queue.append((self.entity_id, waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            # stops the wait, or passes on a slot given in the meantime
            self.release()
            raise

    def release(self):
        """Give back the slot of the entity, or stop its wait for one.

        The slot goes straight to the entity waiting the longest. It does
        nothing if the entity neither holds nor waits for a slot.
        """
        with self._lock:
            for waiting in self._queue:
                if waiting[0] == self.entity_id:
                    self._queue.remove(waiting)
                    return
            if self.entity_id not in self.holders:
                return
            self.holders.remove(self.entity_id)
            while self._queue:
                entity_id, waiter = self._queue.popleft()
                try:
                    waiter.get_loop().call_soon_threadsafe(_wake, waiter)
                except RuntimeError:  # the loop of a stopped entity is closed
                    continue
                self.holders.add(entity_id)
                return


def _wake(waiter: asyncio.Future):
    """Wake an entity given a slot, unless it stopped waiting.

    Args:
        waiter (asyncio.Future): The future the entity is waiting on.
    """
    if not waiter.done():
        waiter.set_result(None)
