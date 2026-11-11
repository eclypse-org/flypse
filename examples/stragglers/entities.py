"""The entities of the stragglers example.

The clients train locally, send their model to the server and wait for the
next global one. The server closes a round at a deadline, with whichever models
arrived in time.
"""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING

from flypse.data import FractionSplitter
from flypse.federation import (
    Entity,
    Federation,
    get_star,
)

from examples.stragglers.model import (
    FedAvgModel,
    LinearModel,
    local_train,
    mean_squared_error,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from flypse.federation import Message


class Client(Entity):
    """A client of the federation.

    On every step it trains the received global model on its local data, sends
    the local model to the server, and waits for the next global model,
    evaluating it on its test part every few rounds.
    """

    def __init__(
        self,
        entity_id: str,
        server_id: str,
        *,
        n_features: int,
        epochs: int,
        lr: float,
        eval_every: int,
    ):
        """Initialise the client.

        Args:
            entity_id (str): The identifier of the client.
            server_id (str): The identifier of the server.
            n_features (int): The number of features of the model.
            epochs (int): The number of local passes over the data.
            lr (float): The learning rate.
            eval_every (int): The number of rounds between two evaluations.
        """
        super().__init__(entity_id, store_step=True)
        self._server_id = server_id
        self._n_features = n_features
        self._epochs = epochs
        self._lr = lr
        self._eval_every = eval_every
        self.model = FedAvgModel(LinearModel([0.0] * n_features, 0.0))

    def on_deploy(self):
        """Bind the local data, initialise the model and split off a test part.

        The initial model and the split are drawn from the ``rng`` of the
        entity, seeded from the seed of the process. The entity trains on 80%
        of its data and keeps the rest to evaluate the model.
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

    async def step(self):
        """Train locally and synchronise with the server.

        Returns:
            float | None: The local loss, or None if no data is bound.
        """
        if self.data is None:
            return None

        self.model.model = local_train(
            self.model.model,
            self.data,
            epochs=self._epochs,
            lr=self._lr,
        )
        record = self.model.update_version(self.model.get_parameters(), "local")
        loss = mean_squared_error(self.model.model, self.data)

        await self.send(
            self._server_id,
            record,
            n_samples=len(self.data),
            loss=loss,
        )
        message = await self.receive()
        if message.record.version[0] % self._eval_every == 0:
            self.run_evaluation(message.record, "test")
        self.model.load_version(message.record)
        return loss


class DeadlineServer(Entity):
    """A server that does not wait for its slowest clients.

    A round ends when every client has answered, or at the deadline. A model
    generated for a previous round is late: it does not take part in the average.
    Everyone who wrote gets the latest model back, on time or not, so that every
    client always has one request pending, and the clients never need to know
    about the deadline.
    """

    def __init__(self, entity_id: str, *, n_features: int, deadline: float):
        """Initialise the server.

        Args:
            entity_id (str): The identifier of the server.
            n_features (int): The number of features of the model.
            deadline (float): The duration of a round, at most, in seconds.
        """
        super().__init__(entity_id, with_data=False)
        self._deadline = deadline
        self.model = FedAvgModel(LinearModel([0.0] * n_features, 0.0))

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
            Client(
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
