"""The entities of the hierarchical federated averaging example.

The clients are the ones of the federated averaging example, and the root is
its aggregator: only the group aggregators in between are new. They average the
models of their clients at every round, and exchange their own with the root
every few rounds, so that a version of the model tells where it comes from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flypse.federation import (
    Federation,
    get_tree,
)

from examples.fedavg.entities import (
    FedAvgClient,
    FedAvgServer,
    weighted_average,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

VERSION_LEVELS = ("global", "group", "local")
"""The version levels of the models, from the coarsest to the finest."""


class GroupAggregator(FedAvgServer):
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
        """Initialize the group aggregator.

        Args:
            entity_id (str): The identifier of the aggregator.
            root_id (str): The identifier of the root.
            client_ids (Sequence[str]): The identifiers of its clients.
            n_features (int): The number of features of the model.
            global_every (int): The number of group rounds per global round.
        """
        super().__init__(
            entity_id, n_features=n_features, version_levels=VERSION_LEVELS
        )
        self._root_id = root_id
        self._client_ids = list(client_ids)
        self._global_every = global_every
        self._rounds = 0

    async def step(self):
        """Aggregate the clients, and synchronize with the root when it is time."""
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
            FedAvgClient(
                client_id,
                group_id,
                n_features=n_features,
                epochs=epochs,
                lr=lr,
                eval_every=eval_every,
                version_levels=VERSION_LEVELS,
            )
            for client_id in client_ids
        ]
    return get_tree(
        FedAvgServer(root_id, n_features=n_features, version_levels=VERSION_LEVELS),
        children,
        federation_id="HierarchicalFedAvg",
    )
