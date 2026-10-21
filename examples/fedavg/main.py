"""Run a basic federated averaging experiment on a synthetic regression task.

Run from the repository root with::

    uv run python -m examples.fedavg.main

A star federation of learners around one aggregator is emulated with a mirror
infrastructure (one device per entity, resources equal to the requirements).
Every learner trains a linear model on its own (non-IID) slice of the data and
exchanges it with the aggregator, which averages the models (FedAvg) and
broadcasts the global one back.

Ray starts its workers through ``uv run`` when the driver runs under it,
syncing a fresh environment for each worker: the switch below, set before Ray
is imported, makes the workers reuse this environment instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from flypse.data import UniformDataset
from flypse.process import FederatedProcess
from flypse.system import SystemConfig

from examples.fedavg.data import make_source
from examples.fedavg.entities import build_federation

SERVER_ID = "server"
"""The identifier of the aggregator."""

CLIENT_IDS = [f"client_{i}" for i in range(5)]
"""The identifiers of the learners."""

N_FEATURES = 1
"""The number of features of the linear model."""

N_SAMPLES = 300
"""The total number of samples of the federation."""

EPOCHS = 5
"""The number of local passes over the data per round."""

LR = 0.02
"""The learning rate of the local SGD."""

STEP_EVERY_MS = 100.0
"""How often the simulation enacts the placement (and deploys the entities)."""

SECONDS = 10.0
"""How long the emulation runs."""


def main() -> None:
    """Build and run the federated averaging experiment."""
    source = make_source(N_SAMPLES, noise=0.1, seed=0)
    data = UniformDataset(source)
    federation = build_federation(
        CLIENT_IDS,
        SERVER_ID,
        n_features=N_FEATURES,
        epochs=EPOCHS,
        lr=LR,
    )

    process = FederatedProcess(
        federation,
        data,
        SystemConfig(),
        seed=0,
        step_every_ms=STEP_EVERY_MS,
        include_default_metrics=True,
    )
    process.run(seconds=SECONDS)

    print(process.report.service(service_ids=SERVER_ID))


if __name__ == "__main__":
    main()
