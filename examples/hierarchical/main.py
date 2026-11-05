"""Run a hierarchical federated averaging experiment.

Run from the repository root with::

    uv run python -m examples.hierarchical.main

A tree federation is emulated with a mirror infrastructure: four clients in two
groups, two group aggregators and a root. The groups average their clients at
every round and the root averages the groups every few rounds, so the models of
the two groups differ until the next global round. Every client evaluates every
model it receives, which shows it: the table counts the distinct models sharing
each version.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from flypse.data import UniformDataset
from flypse.process import FederatedProcess

from examples.fedavg.data import make_source
from examples.hierarchical.entities import build_federation

ROOT_ID = "root"
"""The identifier of the root."""

GROUPS = {
    "edge_0": ["client_0", "client_1"],
    "edge_1": ["client_2", "client_3"],
}
"""The identifiers of the clients of each group aggregator."""

N_FEATURES = 1
"""The number of features of the linear model."""

N_SAMPLES = 240
"""The total number of samples of the federation."""

EPOCHS = 5
"""The number of local passes over the data per round."""

LR = 0.02
"""The learning rate of the local SGD."""

GLOBAL_EVERY = 3
"""The number of group rounds per global round."""

EVAL_EVERY = 1
"""The number of global rounds between two evaluations."""

STEP_EVERY_MS = 100.0
"""How often the simulation enacts the placement (and deploys the entities)."""

SECONDS = 10.0
"""How long the emulation runs."""

N_ROWS = 7
"""The number of versions shown from the start of the run."""


def main() -> None:
    """Build and run the hierarchical federated averaging experiment."""
    federation = build_federation(
        GROUPS,
        ROOT_ID,
        n_features=N_FEATURES,
        epochs=EPOCHS,
        lr=LR,
        eval_every=EVAL_EVERY,
        global_every=GLOBAL_EVERY,
    )
    data = UniformDataset(make_source(N_SAMPLES, noise=0.1, seed=0))

    process = FederatedProcess(
        federation,
        data,
        seed=0,
        step_every_ms=STEP_EVERY_MS,
    )
    process.run(seconds=SECONDS)

    by_version = defaultdict(list)
    for record in process.reported:
        by_version[tuple(record["version"])].append(record)

    print(f"{'version':<14}{'evaluations':<13}{'models':<8}{'mean test loss'}")
    for version in sorted(by_version)[:N_ROWS]:
        records = by_version[version]
        n_models = len({record["model"] for record in records})
        loss = mean(record["metrics"]["loss"] for record in records)
        print(f"{version!s:<14}{len(records):<13}{n_models:<8}{loss:.6f}")


if __name__ == "__main__":
    main()
