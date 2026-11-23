# Compose an algorithm

An algorithm is one choice per decision of a round. This page shows how the choices come
together in the `step` of an entity, for the protocols most of the others derive from.

| Algorithm | Screening | Weighting | Aggregation | Optimisation |
| --- | --- | --- | --- | --- |
| FedAvg | | `by_samples` | `mean` of the models | adopt |
| FedAvgM | | `by_samples` | `mean` of the updates | `SGD(momentum=0.9)` |
| FedAdam, FedYogi, FedAdagrad | | `by_samples` | `mean` of the updates | `Adam`, `Yogi`, `Adagrad(beta1=0)` |
| FedExP | | equal | | `fedexp` |
| FedADP | | `by_samples` and `FedADP` | `mean` of the updates | `SGD()` |
| FedMGDA+ | | `min_norm` | `mean` of the normalised updates | `SGD()` |
| FedAsync | | `polynomial_staleness` | | `interpolate` |
| FedBuff | | `polynomial_staleness` | `mean` of the updates, scaled by their staleness | `SGD()` |
| Multi-Krum | `krum` | equal | `mean` | adopt |
| Bulyan | `bulyan` | | `meamed` | adopt |
| RFA | | `by_samples` | `geometric_median` | adopt |
| Clustered federated learning | `cosine_bipartition` | `by_samples` | `mean` of each cluster | adopt |
| FedAMP | | `fedamp` | `mean` for each recipient | the client trains towards it |
| DiLoCo | | equal | `mean` of the updates | `SGD(lr=0.7, momentum=0.9, nesterov=True)` |
| Decentralised averaging | | `metropolis_hastings` | `mean` with the neighbours | adopt |

On this page, every client answers whoever sent it a model. `MeanModel` is the model of the
[first federation](../getting-started/first-federation.md), and `train` stands for your own
local training:

```python
import numpy as np

from flypse.federation import Entity
from flypse.strategies import aggregation, optimization, screening, selection, weighting


class Client(Entity):
    """Trains the model it is sent, and answers with the result."""

    def __init__(self, entity_id):
        super().__init__(entity_id)
        self.model = MeanModel({"mean": np.zeros(1)})

    async def step(self):
        message = await self.receive()
        self.model.load_version(message.record)
        train(self.model.model, self.data)
        record = self.model.update_version(self.model.get_parameters(), "local")
        await self.send(message.sender, record, n_samples=len(self.data))
```

## A synchronous server

The server of FedAdam selects some clients, sends them its model, and waits for all of
their answers:

```python
class FedAdamServer(Entity):
    def __init__(self, entity_id, k):
        super().__init__(entity_id, with_data=False)
        self.k = k
        self.model = MeanModel({"mean": np.zeros(1)})
        self.optimizer = optimization.Adam(lr=1.0)

    async def step(self):
        selected, _ = selection.uniform(await self.get_neighbors(), self.k, self.rng)
        await self.send(selected, self.model.versions[-1])
        messages = [await self.receive(source=client) for client in selected]

        own = self.model.get_parameters()
        deltas = [m.record.parameters - own for m in messages]
        weights = weighting.normalise(weighting.by_samples(messages))
        delta = aggregation.mean(deltas, weights)
        self.model.update_version(self.optimizer.step(own, delta), "global")
```

Replacing `optimization.Adam` gives the other optimisers of the table, and passing
`own + delta` to `update_version` gives federated averaging.

## A buffered server

An asynchronous server never waits for a given client. The one of FedBuff updates its model
every `k` contributions, whoever they come from, and sends the result to those who
contributed, so that every client always has one model to train:

```python
class FedBuffServer(Entity):
    def __init__(self, entity_id, k):
        super().__init__(entity_id, with_data=False)
        self.k = k
        self.model = MeanModel({"mean": np.zeros(1)})
        self.optimizer = optimization.SGD()
        self.started = False

    def on_deploy(self):
        super().on_deploy()
        self.started = False

    async def step(self):
        if not self.started:
            self.started = True
            await self.bcast(self.model.versions[-1])
        messages = [await self.receive() for _ in range(self.k)]

        lags = [self.model.lag(m.record, "global") for m in messages]
        bases = [self.model.base(m.record, "global") for m in messages]
        deltas = [
            m.record.parameters - base.parameters
            for m, base in zip(messages, bases, strict=True)
        ]
        factors = weighting.polynomial_staleness(lags)
        scaled = [s * d for s, d in zip(factors, deltas, strict=True)]
        new = self.optimizer.step(
            self.model.get_parameters(), aggregation.mean(scaled)
        )
        record = self.model.update_version(new, "global")
        await self.send([m.sender for m in messages], record)
```

A contribution may have been trained on an older model than the current one. Its
[version](../concepts/versions.md#base-and-lag) tells which one, so
[base][flypse.federation.model.FederatedModel.base] finds the model its update is relative
to, and [lag][flypse.federation.model.FederatedModel.lag] how stale it is. The
factors are not normalised, as a buffer of stale updates must move the model less.

The server opens the protocol, so it does it again when it is
[deployed again](../concepts/system.md#placement-reset): the messages in flight are lost, and
its clients are waiting. `k` must not exceed the number of clients.

!!! info "Important"

    The base of a contribution must still be in the
    [history](../concepts/versions.md#the-history) of the server: a `max_versions` not
    larger than the largest staleness makes `base` raise `ValueError`.

## A Byzantine-robust server

A server that does not trust every client screens their updates before combining them.
This is the synchronous server above, with Multi-Krum:

```python
class MultiKrumServer(Entity):
    def __init__(self, entity_id, f, m):
        super().__init__(entity_id, with_data=False)
        self.f = f
        self.m = m
        self.model = MeanModel({"mean": np.zeros(1)})

    async def step(self):
        clients = sorted(await self.get_neighbors())
        await self.send(clients, self.model.versions[-1])
        messages = [await self.receive(source=client) for client in clients]

        own = self.model.get_parameters()
        deltas = [m.record.parameters - own for m in messages]
        kept = screening.krum(deltas, f=self.f, m=self.m)
        new = own + aggregation.mean([deltas[i] for i in kept])
        self.model.update_version(new, "global")
        self.report(round=self.model.version[0], kept=kept, estimate=new[0])
```

With seven clients over the shuffled integers from 0 to 99, one of which always answers
with the model it was sent raised by 1000, the estimate of the server goes:

| Server | Round 1 | Round 2 | Round 3 | Settles at |
| --- | --- | --- | --- | --- |
| The mean of all the updates | 164.0 | 257.7 | 311.3 | 382.7 |
| `MultiKrumServer("server", f=1, m=3)` | 25.8 | 38.7 | 45.2 | 51.6 |

The mean of the data of the honest clients is 49.1. Multi-Krum is not far, and not exact: it
settles at the mean of the three clients it keeps, and the ones it drops are mostly honest.

## A peer

With no server, every entity trains and aggregates. The mixing row of a peer comes from
its connections:

```python
class Peer(Entity):
    def __init__(self, entity_id):
        super().__init__(entity_id)
        self.model = MeanModel({"mean": np.zeros(1)})

    async def step(self):
        train(self.model.model, self.data)
        record = self.model.update_version(self.model.get_parameters(), "local")

        own, row = weighting.metropolis_hastings(await self.get_connections())
        await self.bcast(record)
        peers = sorted(row)
        messages = [await self.receive(source=peer) for peer in peers]

        vectors = [record.parameters, *(m.record.parameters for m in messages)]
        weights = np.array([own, *(row[peer] for peer in peers)])
        self.model.update_version(aggregation.mean(vectors, weights), "global")
```

A peer reads the records it receives and never loads them: its `"global"` version counts
its own rounds. Receiving from each neighbour in turn keeps a round to its own messages, as a
neighbour may already be one ahead.

!!! note "See also"

    - [Strategies](index.md)
    - [Federated averaging](../examples/fedavg.md), the same round written by hand
    - [Decentralised federated averaging](../examples/decentralized.md)
