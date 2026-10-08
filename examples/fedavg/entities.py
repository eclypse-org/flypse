"""The entities of the federated averaging example.

The synchronization protocol is *not* part of FLYPSE core: it emerges entirely
from the implementation of the entities' ``step`` methods. Learners train
locally and exchange their models with the aggregator; the aggregator averages
them and broadcasts the result back.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from flypse.federation import (
    Entity,
    Federation,
)

from examples.fedavg.model import (
    LinearModel,
    federated_average,
    local_train,
    mean_squared_error,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class FedAvgClient(Entity):
    """A learner of the federation.

    On every step it trains the received global model on its local data, sends
    the local model to the aggregator, and waits for the next global model.
    """

    def __init__(
        self,
        entity_id: str,
        server_id: str,
        *,
        n_features: int,
        epochs: int,
        lr: float,
        seed: int | None = None,
        store_step: bool = True,
    ):
        """Initialize the learner.

        Args:
            entity_id (str): The identifier of the learner.
            server_id (str): The identifier of the aggregator.
            n_features (int): The number of features of the model.
            epochs (int): The number of local passes over the data.
            lr (float): The learning rate.
            seed (int | None): The seed of the initial model. Defaults to None.
            store_step (bool): Whether the local loss is stored for reporting.
                Defaults to True.
        """
        super().__init__(entity_id, store_step=store_step)
        self._server_id = server_id
        self._n_features = n_features
        self._epochs = epochs
        self._lr = lr
        self._rng = random.Random(seed)
        self._model = LinearModel([0.0] * n_features, 0.0)

    def on_deploy(self):
        """Bind the local data and initialize the model randomly."""
        super().on_deploy()
        self._model = LinearModel(
            [self._rng.uniform(-0.1, 0.1) for _ in range(self._n_features)],
            self._rng.uniform(-0.1, 0.1),
        )

    async def step(self):
        """Train locally and synchronize with the aggregator.

        Returns:
            float | None: The local loss, or None if no data is bound.
        """
        if self.data is None:
            return None

        self._model = local_train(
            self._model,
            self.data,
            epochs=self._epochs,
            lr=self._lr,
        )
        loss = mean_squared_error(self._model, self.data)

        await self.mpi.send(
            self._server_id,
            {
                "weights": self._model.weights,
                "bias": self._model.bias,
                "n_samples": len(self.data),
                "loss": loss,
            },
        )
        message = await self.mpi.recv()
        self._model = LinearModel(message["weights"], message["bias"])
        return loss


class FedAvgServer(Entity):
    """The aggregator of the federation.

    On every step it collects the local models of all its neighbors, averages
    them (weighted by the number of samples), and broadcasts the global model
    back to them.
    """

    def __init__(self, entity_id: str, *, store_step: bool = True):
        """Initialize the aggregator.

        Args:
            entity_id (str): The identifier of the aggregator.
            store_step (bool): Whether the global loss is stored for
                reporting. Defaults to True.
        """
        super().__init__(entity_id, store_step=store_step)
        self._model: LinearModel | None = None

    async def step(self):
        """Aggregate the local models of the neighbors and broadcast them.

        Returns:
            float | None: The weighted average of the local losses, or None if
            there are no neighbors.
        """
        neighbors = await self.mpi.get_neighbors()
        updates: list[tuple[LinearModel, int]] = []
        losses: list[tuple[float, int]] = []
        for _ in neighbors:
            message = await self.mpi.recv()
            updates.append(
                (
                    LinearModel(message["weights"], message["bias"]),
                    message["n_samples"],
                )
            )
            losses.append((message["loss"], message["n_samples"]))

        if not updates:
            return None

        self._model = federated_average(updates)
        await self.mpi.bcast(
            {"weights": self._model.weights, "bias": self._model.bias}
        )

        n_total = sum(n for _, n in losses)
        global_loss = sum(loss * n for loss, n in losses) / n_total
        self.logger.info(f"FedAvg round loss: {global_loss:.6f}")
        return global_loss


def build_federation(
    client_ids: Sequence[str],
    server_id: str,
    *,
    n_features: int,
    epochs: int,
    lr: float,
    seed: int | None = None,
) -> Federation:
    """Build the star-shaped federation of the example.

    Args:
        client_ids (Sequence[str]): The identifiers of the learners.
        server_id (str): The identifier of the aggregator.
        n_features (int): The number of features of the model.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.
        seed (int | None): The seed of the learners. Defaults to None.

    Returns:
        Federation: The federation of the example.
    """
    federation = Federation("FedAvg")
    federation.add_entity(FedAvgServer(server_id))
    for i, client_id in enumerate(client_ids):
        federation.add_entity(
            FedAvgClient(
                client_id,
                server_id,
                n_features=n_features,
                epochs=epochs,
                lr=lr,
                seed=None if seed is None else seed + i,
            )
        )
        federation.add_connection(server_id, client_id)
    return federation
