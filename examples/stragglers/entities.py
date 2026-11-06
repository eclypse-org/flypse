"""The entities of the stragglers example.

The clients are the ones of the federated averaging example: only the server
changes. It closes a round at a deadline, with whichever models arrived in time.
"""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING

from flypse.federation import (
    Federation,
    get_star,
)

from examples.fedavg.entities import (
    FedAvgClient,
    FedAvgServer,
    weighted_average,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class DeadlineServer(FedAvgServer):
    """A server that does not wait for its slowest clients.

    A round ends when every client has answered, or at the deadline. A model
    generated for a previous round is late: it does not take part in the average.
    Everyone who wrote gets the latest model back, on time or not, so that every
    client always has one request pending, and the clients need no changes.
    """

    def __init__(self, entity_id: str, *, n_features: int, deadline: float):
        """Initialise the server.

        Args:
            entity_id (str): The identifier of the server.
            n_features (int): The number of features of the model.
            deadline (float): The duration of a round, at most, in seconds.
        """
        super().__init__(entity_id, n_features=n_features)
        self._deadline = deadline

    async def _collect(self, n_clients: int) -> list:
        """Collect the messages arriving before the deadline of the round.

        Args:
            n_clients (int): The number of clients of the federation.

        Returns:
            list: The messages of the clients, on time or not.
        """
        round_ = self.model.version[0]
        deadline = monotonic() + self._deadline
        messages = []
        on_time = 0
        while on_time < n_clients:
            try:
                message = await self.receive(timeout=deadline - monotonic())
            except TimeoutError:
                break
            messages.append(message)
            on_time += message.record.version[0] == round_
        return messages

    async def step(self):
        """Run a round: collect until the deadline, aggregate and reply."""
        started = monotonic()
        round_ = self.model.version[0]
        messages = await self._collect(len(await self.get_neighbors()))
        on_time = [m for m in messages if m.record.version[0] == round_]

        # with nobody on time the model stays the same, and the clients get it back
        record = self.model.versions[-1]
        if on_time:
            record = self.model.update_version(weighted_average(on_time), "global")
        if messages:
            await self.send([m.sender for m in messages], record)

        self.report(
            round=round_,
            participants=len(on_time),
            late=len(messages) - len(on_time),
            seconds=monotonic() - started,
        )


def build_federation(
    client_ids: Sequence[str],
    server_id: str,
    *,
    n_features: int,
    epochs: int,
    lr: float,
    eval_every: int,
    deadline: float,
) -> Federation:
    """Build the star-shaped federation of the example.

    Args:
        client_ids (Sequence[str]): The identifiers of the clients.
        server_id (str): The identifier of the server.
        n_features (int): The number of features of the model.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.
        eval_every (int): The number of rounds between two evaluations.
        deadline (float): The duration of a round, at most, in seconds.

    Returns:
        Federation: The federation of the example.
    """
    return get_star(
        DeadlineServer(server_id, n_features=n_features, deadline=deadline),
        [
            FedAvgClient(
                client_id,
                server_id,
                n_features=n_features,
                epochs=epochs,
                lr=lr,
                eval_every=eval_every,
            )
            for client_id in client_ids
        ],
        federation_id="Stragglers",
    )
