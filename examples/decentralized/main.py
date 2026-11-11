"""Run a decentralised federated averaging experiment.

Run from the repository root with::

    uv run python -m examples.decentralized.main

A ring of peers is emulated with a mirror infrastructure, with no server: each
peer averages its model with the ones of its two neighbours. Every few rounds
the peers evaluate their model on their own test part and report it, so the
table shows the mean test loss and how far apart the models of the peers are.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import dist
from statistics import mean

from flypse.data import UniformDataset
from flypse.process import FederatedProcess

from examples.decentralized.data import make_source
from examples.decentralized.entities import build_federation

PEER_IDS = [f"peer_{i}" for i in range(5)]
"""The identifiers of the peers, in ring order."""

N_FEATURES = 1
"""The number of features of the linear model."""

N_SAMPLES = 300
"""The total number of samples of the federation."""

EPOCHS = 1
"""The number of local passes over the data per round."""

LR = 0.01
"""The learning rate of the local SGD."""

EVAL_EVERY = 10
"""The number of rounds between two evaluations."""

STEP_EVERY_MS = 100.0
"""How often the process takes a step; the first one deploys the entities."""

SECONDS = 10.0
"""How long the emulation runs."""

N_ROWS = 6
"""The number of rounds shown from the start of the run."""


def main() -> None:
    """Build and run the decentralised federated averaging experiment."""
    federation = build_federation(
        PEER_IDS,
        n_features=N_FEATURES,
        epochs=EPOCHS,
        lr=LR,
        eval_every=EVAL_EVERY,
    )
    data = UniformDataset(make_source(N_SAMPLES, noise=0.1, seed=0))

    process = FederatedProcess(
        federation,
        data,
        seed=0,
        step_every_ms=STEP_EVERY_MS,
    )
    process.run(seconds=SECONDS)

    losses, models = defaultdict(list), defaultdict(list)
    for record in process.reported:
        if "metrics" in record:
            losses[record["round"]].append(record["metrics"]["loss"])
        else:
            models[record["round"]].append(record["parameters"])

    complete = [round_ for round_ in sorted(losses) if len(models[round_]) == len(PEER_IDS)]
    print(f"{'round':<8}{'mean test loss':<17}{'largest distance between peers'}")
    for round_ in complete[:N_ROWS]:
        spread = max(dist(a, b) for a, b in combinations(models[round_], 2))
        print(f"{round_:<8}{mean(losses[round_]):<17.6f}{spread:.6f}")


if __name__ == "__main__":
    main()
