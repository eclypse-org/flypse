"""The entities of the decentralised federated averaging example.

There is no server: every peer trains on its data, exchanges its model with its
neighbours and averages what it gets back. The models of the peers reach a
consensus through repeated averaging, as a model crosses the ring one hop at a
time.
"""

from __future__ import annotations

from collections import (
    defaultdict,
    deque,
)
from typing import TYPE_CHECKING

from flypse.data import FractionSplitter
from flypse.federation import (
    Entity,
    Federation,
    Message,
    get_ring,
)

from examples.decentralized.model import (
    FedAvgModel,
    LinearModel,
    local_train,
    mean_squared_error,
)

if TYPE_CHECKING:
    import numpy as np


class Peer(Entity):
    """A peer of the federation, both a learner and an aggregator.

    On every step it trains locally, sends its model to its neighbours, and
    averages the models they send for the same round. A neighbour can be a
    round ahead of another, so messages wait in a queue per sender until their
    turn comes, keeping the average of a round made of that round only.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        n_features: int,
        epochs: int,
        lr: float,
        eval_every: int,
    ):
        """Initialise the peer.

        Args:
            entity_id (str): The identifier of the peer.
            n_features (int): The number of features of the model.
            epochs (int): The number of local passes over the data.
            lr (float): The learning rate.
            eval_every (int): The number of rounds between two evaluations.
        """
        super().__init__(entity_id)
        self._n_features = n_features
        self._epochs = epochs
        self._lr = lr
        self._eval_every = eval_every
        self._inbox: dict[str, deque[Message]] = defaultdict(deque)
        self.model = FedAvgModel(LinearModel([0.0] * n_features, 0.0))

    def on_deploy(self):
        """Bind the local data, initialise the model and split off a test part.

        The initial model and the split are drawn from the ``rng`` of the
        entity, seeded from the seed of the process. The peer trains on 80% of
        its data and keeps the rest to evaluate the model.
        """
        super().on_deploy()
        self.model.model = LinearModel(
            [self.rng.uniform(-0.1, 0.1) for _ in range(self._n_features)],
            self.rng.uniform(-0.1, 0.1),
        )
        if self.data is not None:
            splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
            self.splits = splitter.apply(self.data)
            self.data = self.splits["train"]

    def evaluate(self, record, data):
        """Evaluate a version of the model on the given data.

        Args:
            record (ModelVersion): The version of the model to evaluate.
            data (DataView): The data to evaluate it on.

        Returns:
            dict[str, float]: The loss of the version on the data.
        """
        return {"loss": mean_squared_error(self.model.snapshot(record), data)}

    async def _receive_from(self, sender: str) -> Message:
        """Receive the next message of a sender, keeping the others aside.

        Args:
            sender (str): The identifier of the sender.

        Returns:
            Message: The oldest message of the sender not yet received.
        """
        while not self._inbox[sender]:
            message = await self.receive()
            self._inbox[message.sender].append(message)
        return self._inbox[sender].popleft()

    async def step(self):
        """Train locally and average with the neighbours."""
        self.model.model = local_train(
            self.model.model,
            self.data,
            epochs=self._epochs,
            lr=self._lr,
        )
        n_samples = len(self.data)
        record = self.model.update_version(self.model.get_parameters(), "local")

        neighbours = sorted(await self.get_neighbors())
        await self.send(neighbours, record, n_samples=n_samples)
        messages = [await self._receive_from(peer) for peer in neighbours]
        messages.append(Message(self.id, record, {"n_samples": n_samples}))

        # a peer does not load what it receives: its version is its own round
        record = self.model.update_version(weighted_average(messages), "global")
        if record.version[0] % self._eval_every == 0:
            self.run_evaluation(record, "test")
            self.report(round=record.version[0], parameters=record.parameters)


def weighted_average(messages: list[Message]) -> np.ndarray:
    """Average the models of the messages, weighted by their number of samples.

    Args:
        messages (list[Message]): The messages, sorted by sender in place.

    Returns:
        np.ndarray: The average model: Θ = Σ_c (n_c / n) Θ_c.
    """
    # a fixed order: float rounding would make the model follow the arrivals
    messages.sort(key=lambda message: message.sender)
    n_total = sum(m.data["n_samples"] for m in messages)
    return sum(m.record.parameters * m.data["n_samples"] for m in messages) / n_total


def build_federation(
    peer_ids: list[str],
    *,
    n_features: int,
    epochs: int,
    lr: float,
    eval_every: int,
) -> Federation:
    """Build the ring-shaped federation of the example.

    Args:
        peer_ids (list[str]): The identifiers of the peers, in ring order.
        n_features (int): The number of features of the model.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.
        eval_every (int): The number of rounds between two evaluations.

    Returns:
        Federation: The federation of the example.
    """
    return get_ring(
        [
            Peer(
                peer_id,
                n_features=n_features,
                epochs=epochs,
                lr=lr,
                eval_every=eval_every,
            )
            for peer_id in peer_ids
        ],
        federation_id="DecentralizedFedAvg",
    )
