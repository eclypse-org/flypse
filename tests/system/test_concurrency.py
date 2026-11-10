"""Tests for the ConcurrencyHandler class."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from flypse.system import ConcurrencyHandler


def _bound(handler: ConcurrencyHandler, *entity_ids: str) -> list[ConcurrencyHandler]:
    """Return the handler as used by each of the given entities."""
    return [handler.bind(entity_id) for entity_id in entity_ids]


async def _queued(*bound: ConcurrencyHandler) -> list[asyncio.Task]:
    """Make the given entities ask for a slot, in order, and wait for it."""
    tasks = [asyncio.create_task(b.acquire()) for b in bound]
    await asyncio.sleep(0)
    return tasks


async def test_handler_admits_up_to_its_limit():
    """Entities beyond max_concurrency wait until a slot is given back."""
    a, b, c = _bound(ConcurrencyHandler(2), "a", "b", "c")
    await a.acquire()
    await b.acquire()
    (waiting,) = await _queued(c)

    assert not waiting.done()
    a.release()
    await waiting
    assert a.holders == {"b", "c"}


async def test_handler_without_limit_never_waits():
    """With no limit, every entity holds a slot at once."""
    handler = ConcurrencyHandler()
    for bound in _bound(handler, *"abcde"):
        await bound.acquire()

    assert handler.holders == set("abcde")


async def test_handler_gives_slots_in_the_order_asked():
    """A slot goes to the entity waiting the longest, even if its holder asks again."""
    a, b, c = _bound(ConcurrencyHandler(1), "a", "b", "c")
    await a.acquire()
    waiting = await _queued(b, c)
    a.release()
    waiting += await _queued(a)

    for holder in (b, c):
        assert a.holders == {holder.entity_id}
        holder.release()
    await asyncio.gather(*waiting)
    assert a.holders == {"a"}


async def test_a_cancelled_wait_gives_up_its_turn():
    """An entity that stops waiting is not given a slot."""
    a, b, c = _bound(ConcurrencyHandler(1), "a", "b", "c")
    await a.acquire()
    cancelled, waiting = await _queued(b, c)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    a.release()
    await waiting
    assert a.holders == {"c"}


async def test_a_slot_given_to_a_cancelled_wait_is_passed_on():
    """A slot given to an entity cancelled before it woke up goes to the next."""
    a, b, c = _bound(ConcurrencyHandler(1), "a", "b", "c")
    await a.acquire()
    cancelled, waiting = await _queued(b, c)
    a.release()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    await waiting
    assert a.holders == {"c"}


async def test_a_slot_skips_entities_whose_loop_is_closed():
    """A slot is not given to an entity stopped while waiting, but to the next."""
    handler = ConcurrencyHandler(1)
    a, c = _bound(handler, "a", "c")
    await a.acquire()
    stopped = asyncio.new_event_loop()
    handler._queue.append(("b", stopped.create_future()))
    stopped.close()
    (waiting,) = await _queued(c)

    a.release()
    await waiting
    assert handler.holders == {"c"}


def test_a_release_wakes_an_entity_on_another_thread():
    """A slot given back on one thread wakes an entity waiting on another one."""
    a, b = _bound(ConcurrencyHandler(1), "a", "b")
    asyncio.run(a.acquire())
    waiting = threading.Thread(target=asyncio.run, args=(b.acquire(),))
    waiting.start()
    while not a._queue and waiting.is_alive():
        time.sleep(0.001)

    a.release()
    waiting.join(timeout=5)
    assert not waiting.is_alive()
    assert a.holders == {"b"}


async def test_an_entity_cannot_ask_twice():
    """An entity holding, or waiting for, a slot cannot ask for another."""
    a, b = _bound(ConcurrencyHandler(1), "a", "b")
    await a.acquire()
    (waiting,) = await _queued(b)

    for bound in (a, b):
        with pytest.raises(RuntimeError, match="already holds or waits"):
            await bound.acquire()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting


def test_release_without_a_slot_does_nothing():
    """An entity neither holding nor waiting for a slot gives back nothing."""
    (a,) = _bound(ConcurrencyHandler(1), "a")

    a.release()

    assert a.holders == set()


async def test_a_block_holds_a_slot_until_it_ends():
    """A slot is held within the block only, even if the block fails."""
    handler = ConcurrencyHandler(1)
    (a,) = _bound(handler, "a")

    with pytest.raises(ValueError):
        async with a:
            assert handler.holders == {"a"}
            raise ValueError
    assert handler.holders == set()


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_handler_admits_at_least_one_entity(max_concurrency):
    """A device admitting no entity at once is rejected."""
    with pytest.raises(ValueError, match="at least 1"):
        ConcurrencyHandler(max_concurrency)
