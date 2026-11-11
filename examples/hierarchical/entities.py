"""The entities of the hierarchical federated averaging example.

The clients train locally and exchange their models with their group
aggregator. The group aggregators average the models of their clients at every
round, and exchange their own with the root every few rounds, so that a version
of the model tells where it comes from. The root averages the groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flypse.data import FractionSplitter
from flypse.federation import (
    Entity,
    Federation,
    get_tree,
)

from examples.hierarchical.model import (
    FedAvgModel,
    LinearModel,
    local_train,
    mean_squared_error,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

    import numpy as np

    from flypse.federation import Message

VERSION_LEVELS = ("global", "group", "local")
"""The version levels of the models, from the coarsest to the finest."""


class Client(Entity):
    """A client of a group.

    On every step it trains the received model on its local data, sends the
    local model to its group aggregator, and waits for the next model,
    evaluating it on its test part every few global rounds.
    """

    def __init__(
        self,
        entity_id: str,
        aggregator_id: str,
        *,
        n_features: int,
        epochs: int,
        lr: float,
        eval_every: int,
    ):
        """Initialise the client.

        Args:
            entity_id (str): The identifier of the client.
            aggregator_id (str): The identifier of its group aggregator.
            n_features (int): The number of features of the model.
            epochs (int): The number of local passes over the data.
            lr (float): The learning rate.
            eval_every (int): The number of global rounds between two
                evaluations.
        """
        super().__init__(entity_id, store_step=True)
        self._aggregator_id = aggregator_id
        self._n_features = n_features
        self._epochs = epochs
        self._lr = lr
        self._eval_every = eval_every
        self.model = FedAvgModel(
            LinearModel([0.0] * n_features, 0.0), version_levels=VERSION_LEVELS
        )

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
        """Train locally and synchronise with the group aggregator.

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
            self._aggregator_id,
            record,
            n_samples=len(self.data),
            loss=loss,
        )
        message = await self.receive()
        if message.record.version[0] % self._eval_every == 0:
            self.run_evaluation(message.record, "test")
        self.model.load_version(message.record)
        return loss


class GroupAggregator(Entity):
    """An aggregator between the clients of a group and the root.

    On every step it averages the models of its clients. Every few rounds it
    sends the result to the root and waits for the global model, then it sends
    the latest model back to its clients. The parties strictly alternate, so a
    plain ``receive`` always returns what the aggregator is waiting for.
    """

    def __init__(
        self,
        entity_id: str,
        root_id: str,
        client_ids: Sequence[str],
        *,
        n_features: int,
        global_every: int,
    ):
        """Initialise the group aggregator.

        Args:
            entity_id (str): The identifier of the aggregator.
            root_id (str): The identifier of the root.
            client_ids (Sequence[str]): The identifiers of its clients.
            n_features (int): The number of features of the model.
            global_every (int): The number of group rounds per global round.
        """
        super().__init__(entity_id, with_data=False)
        self._root_id = root_id
        self._client_ids = list(client_ids)
        self._global_every = global_every
        self._rounds = 0
        self.model = FedAvgModel(
            LinearModel([0.0] * n_features, 0.0), version_levels=VERSION_LEVELS
        )

    async def step(self):
        """Aggregate the clients, and synchronise with the root when it is time."""
        messages = [await self.receive() for _ in self._client_ids]
        record = self.model.update_version(weighted_average(messages), "group")

        self._rounds += 1
        if self._rounds % self._global_every == 0:
            n_samples = sum(m.data["n_samples"] for m in messages)
            loss = sum(m.data["loss"] * m.data["n_samples"] for m in messages)
            await self.send(
                self._root_id, record, n_samples=n_samples, loss=loss / n_samples
            )
            message = await self.receive()
            record = self.model.load_version(message.record)

        # not a bcast: the neighbours of the aggregator include the root
        await self.send(self._client_ids, record)


class Root(Entity):
    """The root of the federation.

    On every step it collects the models of all the group aggregators, averages
    them (weighted by the number of samples), and broadcasts the global model
    back to them.
    """

    def __init__(self, entity_id: str, *, n_features: int):
        """Initialise the root.

        Args:
            entity_id (str): The identifier of the root.
            n_features (int): The number of features of the model.
        """
        super().__init__(entity_id, store_step=True, with_data=False)
        self.model = FedAvgModel(
            LinearModel([0.0] * n_features, 0.0), version_levels=VERSION_LEVELS
        )

    async def step(self):
        """Aggregate the models of the groups and broadcast the global one.

        Returns:
            float | None: The weighted average of the group losses, or None if
            there are no neighbours.
        """
        messages = [await self.receive() for _ in await self.get_neighbors()]
        if not messages:
            return None

        average = weighted_average(messages)
        await self.bcast(self.model.update_version(average, "global"))

        n_total = sum(m.data["n_samples"] for m in messages)
        global_loss = (
            sum(m.data["loss"] * m.data["n_samples"] for m in messages) / n_total
        )
        self.logger.info(f"Global round loss: {global_loss:.6f}")
        return global_loss


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
    groups: Mapping[str, Sequence[str]],
    root_id: str,
    *,
    n_features: int,
    epochs: int,
    lr: float,
    eval_every: int,
    global_every: int,
) -> Federation:
    """Build the tree-shaped federation of the example.

    Args:
        groups (Mapping[str, Sequence[str]]): The identifiers of the clients of
            each group aggregator, keyed by its identifier.
        root_id (str): The identifier of the root.
        n_features (int): The number of features of the model.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.
        eval_every (int): The number of global rounds between two evaluations.
        global_every (int): The number of group rounds per global round.

    Returns:
        Federation: The federation of the example.
    """
    children = {
        root_id: [
            GroupAggregator(
                group_id,
                root_id,
                client_ids,
                n_features=n_features,
                global_every=global_every,
            )
            for group_id, client_ids in groups.items()
        ],
    }
    for group_id, client_ids in groups.items():
        children[group_id] = [
            Client(
                client_id,
                group_id,
                n_features=n_features,
                epochs=epochs,
                lr=lr,
                eval_every=eval_every,
            )
            for client_id in client_ids
        ]
    return get_tree(
        Root(root_id, n_features=n_features),
        children,
        federation_id="HierarchicalFedAvg",
    )
