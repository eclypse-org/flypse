"""The entities of the federated averaging example.

The synchronization protocol is *not* part of FLYPSE core: it emerges entirely
from the implementation of the entities' ``step`` methods. Learners train
locally and exchange their models with the aggregator; the aggregator averages
them and broadcasts the result back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flypse.data import FractionSplitter
from flypse.federation import (
    Entity,
    Federation,
    get_star,
)

from examples.fedavg.model import (
    FedAvgModel,
    LinearModel,
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
        store_step: bool = True,
    ):
        """Initialize the learner.

        Args:
            entity_id (str): The identifier of the learner.
            server_id (str): The identifier of the aggregator.
            n_features (int): The number of features of the model.
            epochs (int): The number of local passes over the data.
            lr (float): The learning rate.
            store_step (bool): Whether the local loss is stored for reporting.
                Defaults to True.
        """
        super().__init__(entity_id, store_step=store_step)
        self._server_id = server_id
        self._n_features = n_features
        self._epochs = epochs
        self._lr = lr
        self.model = FedAvgModel(LinearModel([0.0] * n_features, 0.0))

    def on_deploy(self):
        """Bind the local data, initialize the model and split off a test part.

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
            parts = splitter.apply(self.data)
            self.data, self.test = parts["train"], parts["test"]

    def evaluate(self):
        """Evaluate the latest recorded model on the test part.

        It runs while the entity keeps stepping, so it evaluates a snapshot of
        the latest recorded version rather than the live model.

        Returns:
            dict[str, float] | None: The test loss, or None if no data is bound.
        """
        if self.data is None:
            return None
        model = self.model.snapshot(self.model.versions[-1])
        return {"test_loss": mean_squared_error(model, self.test)}

    async def step(self):
        """Train locally and synchronize with the aggregator.

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
        self.model.load_version(message.record)
        return loss


class FedAvgServer(Entity):
    """The aggregator of the federation.

    On every step it collects the local models of all its neighbors, averages
    them (weighted by the number of samples), and broadcasts the global model
    back to them.
    """

    def __init__(self, entity_id: str, *, n_features: int, store_step: bool = True):
        """Initialize the aggregator.

        Args:
            entity_id (str): The identifier of the aggregator.
            n_features (int): The number of features of the model.
            store_step (bool): Whether the global loss is stored for
                reporting. Defaults to True.
        """
        super().__init__(entity_id, store_step=store_step, with_data=False)
        self.model = FedAvgModel(LinearModel([0.0] * n_features, 0.0))

    async def step(self):
        """Aggregate the local models of the neighbors and broadcast them.

        Returns:
            float | None: The weighted average of the local losses, or None if
            there are no neighbors.
        """
        messages = [await self.receive() for _ in await self.get_neighbors()]
        if not messages:
            return None

        # FedAvg: Θ = Σ_c (n_c / n) Θ_c
        n_total = sum(m.data["n_samples"] for m in messages)
        average = (
            sum(m.record.parameters * m.data["n_samples"] for m in messages) / n_total
        )
        await self.bcast(self.model.update_version(average, "global"))

        global_loss = (
            sum(m.data["loss"] * m.data["n_samples"] for m in messages) / n_total
        )
        self.logger.info(f"FedAvg round loss: {global_loss:.6f}")
        return global_loss


def build_federation(
    client_ids: Sequence[str],
    server_id: str,
    *,
    n_features: int,
    epochs: int,
    lr: float,
) -> Federation:
    """Build the star-shaped federation of the example.

    Args:
        client_ids (Sequence[str]): The identifiers of the learners.
        server_id (str): The identifier of the aggregator.
        n_features (int): The number of features of the model.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.

    Returns:
        Federation: The federation of the example.
    """
    clients = [
        FedAvgClient(
            client_id,
            server_id,
            n_features=n_features,
            epochs=epochs,
            lr=lr,
        )
        for client_id in client_ids
    ]
    return get_star(
        FedAvgServer(server_id, n_features=n_features),
        clients,
        federation_id="FedAvg",
    )
