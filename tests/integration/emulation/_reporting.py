"""A tiny federation whose entities report on every step, emulated through Ray.

It runs as a script, away from the test process, as Ray must start without the
runtime environment of ``uv run``, and the entities must be pickled by value.
Given an output directory and, optionally, the identifier of an entity that
crashes, it writes in that directory what the process reads back, and how many
reports each entity sent, in a file named after it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from flypse.data import UniformDataset
from flypse.federation import (
    Entity,
    get_star,
)
from flypse.process import FederatedProcess

ENTITY_IDS = ["hub", "a", "b"]
"""The identifiers of the entities of the federation."""


class Reporting(Entity):
    """An entity reporting on every step, until it crashes if it has to."""

    def __init__(self, entity_id: str, output: Path, crashes: bool):
        super().__init__(entity_id)
        self.output = output
        self.crashes = crashes

    async def step(self):
        # fast enough to leave reports on their way when the process stops
        await asyncio.sleep(0.0002)
        self.report(loss=np.float32(0.5))
        if self.crashes and self.n_reports == 20:
            raise RuntimeError("The entity crashed.")

    def on_undeploy(self):
        (self.output / self.id).write_text(str(self.n_reports))
        super().on_undeploy()


def main(output: Path, crashing: str | None):
    """Emulate the federation for a second, and write what it reported.

    Args:
        output (Path): The directory to write to.
        crashing (str | None): The identifier of the entity that crashes.
    """
    hub, *leaves = (
        Reporting(entity_id, output, entity_id == crashing) for entity_id in ENTITY_IDS
    )
    process = FederatedProcess(
        get_star(hub, leaves),
        UniformDataset(list(range(6))),
        seed=0,
        step_every_ms=50.0,
        path=output / "run",
    )
    process.run(seconds=1.0)
    (output / "reported.json").write_text(json.dumps(process.reported))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
