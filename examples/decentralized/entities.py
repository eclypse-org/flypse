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

from flypse.federation import (
    Federation,
    Message,
    get_ring,
)

from examples.fedavg.entities import (
    FedAvgClient,
    weighted_average,
)
from examples.fedavg.model import local_train


class Peer(FedAvgClient):
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
        super().__init__(
            entity_id,
            None,
            n_features=n_features,
            epochs=epochs,
            lr=lr,
            eval_every=eval_every,
        )
        self._inbox: dict[str, deque[Message]] = defaultdict(deque)

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
