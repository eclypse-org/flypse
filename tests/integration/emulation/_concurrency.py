"""A server and three clients sharing a device admitting two of them at once.

It runs as a script, away from the test process, as Ray must start without the
runtime environment of ``uv run``, and the entities must be pickled by value.
Every entity reports, on entering its critical operation, the round and how many
entities hold a slot of the device. Given an output directory, it writes in it
what the process reads back.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from flypse.data import UniformDataset
from flypse.federation import (
    Entity,
    get_star,
)
from flypse.process import FederatedProcess
from flypse.system import SystemConfig

CLIENT_IDS = ["a", "b", "c"]
"""The identifiers of the clients."""


class Server(Entity):
    """A server computing a model in its slot, then waiting for every client."""

    def __init__(self):
        super().__init__("server", with_data=False)
        self.round = 0

    async def step(self):
        async with self.concurrency:
            self.report(round=self.round, holders=len(self.concurrency.holders))
            await asyncio.sleep(0.01)
        await self.bcast(round=self.round)
        for _ in CLIENT_IDS:
            await self.receive()
        self.round += 1


class Client(Entity):
    """A client waiting for the model, then training it in its slot."""

    async def step(self):
        message = await self.receive()
        async with self.concurrency:
            self.report(
                round=message.data["round"], holders=len(self.concurrency.holders)
            )
            await asyncio.sleep(0.05)
        await self.send("server", round=message.data["round"])


def main(output: Path):
    """Emulate the federation for a few seconds, and write what it reported.

    Args:
        output (Path): The directory to write to.
    """
    process = FederatedProcess(
        get_star(Server(), [Client(client_id) for client_id in CLIENT_IDS]),
        UniformDataset(list(range(6))),
        SystemConfig("singleton", max_concurrency=2),
        seed=0,
        step_every_ms=50.0,
        path=output / "run",
    )
    process.run(seconds=3.0)
    (output / "reported.json").write_text(json.dumps(process.reported))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
