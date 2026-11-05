"""Run a federated averaging experiment in which a link slows down.

Run from the repository root with::

    uv run python -m examples.stragglers.main

A star federation is emulated with a mirror infrastructure, whose update policies
slow down the uplink of one client for a few seconds. The aggregator closes each
round at a deadline, so the table shows, second by second, how many rounds it
completes, with how many clients, and how many models arrive late.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean

from eclypse import policies

from flypse.data import UniformDataset
from flypse.process import FederatedProcess
from flypse.system import (
    SystemConfig,
    get_mirror,
    get_mirror_strategy,
)

from examples.fedavg.data import make_source
from examples.stragglers.entities import build_federation

SERVER_ID = "server"
"""The identifier of the aggregator."""

CLIENT_IDS = [f"client_{i}" for i in range(5)]
"""The identifiers of the learners."""

STRAGGLER_ID = "client_0"
"""The identifier of the client whose uplink slows down."""

N_FEATURES = 1
"""The number of features of the linear model."""

N_SAMPLES = 300
"""The total number of samples of the federation."""

EPOCHS = 5
"""The number of local passes over the data per round."""

LR = 0.02
"""The learning rate of the local SGD."""

EVAL_EVERY = 10
"""The number of rounds between two evaluations."""

SLOW_LATENCY_MS = 1000.0
"""The latency of the slow uplink, in milliseconds."""

DEADLINE = 0.2
"""The duration of a round, at most, in seconds."""

STEP_EVERY_MS = 100.0
"""How often the simulation enacts the placement and applies the policies."""

SLOW_FROM, SLOW_UNTIL = 30, 60
"""The simulation steps at which the uplink slows down and recovers."""

SECONDS = 10.0
"""How long the emulation runs."""


def slow_uplink(client_id: str, start: int, end: int) -> list:
    """Create the update policies slowing down the uplink of a client.

    Only the direction towards the aggregator slows down, as a mirror link has
    one edge per direction: this way, the replies of the aggregator take no time.

    Args:
        client_id (str): The identifier of the client.
        start (int): The simulation step at which the uplink slows down.
        end (int): The simulation step at which it recovers.

    Returns:
        list: The update policies of the infrastructure.
    """

    def set_latency(latency: float):
        return policies.degrade.set_value(
            latency,
            edge_assets="latency",
            edge_ids=[(client_id, SERVER_ID)],
        )

    return [
        policies.at(start, set_latency(SLOW_LATENCY_MS)),
        policies.at(end, set_latency(0.0)),
    ]


def main() -> None:
    """Build and run the stragglers experiment."""
    federation = build_federation(
        CLIENT_IDS,
        SERVER_ID,
        n_features=N_FEATURES,
        epochs=EPOCHS,
        lr=LR,
        eval_every=EVAL_EVERY,
        deadline=DEADLINE,
    )
    data = UniformDataset(make_source(N_SAMPLES, noise=0.1, seed=0))
    infrastructure = get_mirror(
        federation,
        update_policies=slow_uplink(STRAGGLER_ID, SLOW_FROM, SLOW_UNTIL),
        seed=0,
    )

    process = FederatedProcess(
        federation,
        data,
        SystemConfig(infrastructure, get_mirror_strategy(federation)),
        seed=0,
        step_every_ms=STEP_EVERY_MS,
    )
    process.run(seconds=SECONDS)

    records = process.reported
    rounds = [record for record in records if "participants" in record]
    start = datetime.fromisoformat(rounds[0]["timestamp"])
    by_second = defaultdict(list)
    for record in rounds:
        elapsed = datetime.fromisoformat(record["timestamp"]) - start
        by_second[int(elapsed.total_seconds())].append(record)

    print(f"{'second':<8}{'rounds':<8}{'mean participants':<19}{'late models'}")
    for second, closed in sorted(by_second.items()):
        participants = mean(record["participants"] for record in closed)
        late = sum(record["late"] for record in closed)
        print(f"{second:<8}{len(closed):<8}{participants:<19.1f}{late}")

    evaluations = [record for record in records if "metrics" in record]
    last_round = max(record["round"] for record in evaluations)
    losses = [r["metrics"]["loss"] for r in evaluations if r["round"] == last_round]
    print(f"\nmean test loss at round {last_round}: {mean(losses):.6f}")


if __name__ == "__main__":
    main()
