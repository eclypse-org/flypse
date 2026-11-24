# Weighting

Weighting decides **how much each contribution counts**. A weight is a product of factors,
such as the number of samples behind a model and how stale it is, normalised to sum to 1.

How much a collaborator counts is decided by whoever aggregates, when it aggregates: an
entity sends facts about its contribution, and never its own weight.

The members of [flypse.strategies.weighting][] come in three kinds:

<div class="grid cards" markdown>

-   **[One weight per contribution](#one-weight-per-contribution)**

    ---

    Factors to multiply, for a server or an aggregator.

-   **[One row per peer](#one-row-per-peer)**

    ---

    The mixing weights of an entity with no server.

-   **[One row per recipient](#one-row-per-recipient)**

    ---

    A model of its own for every client.

</div>

## One weight per contribution

| Strategy | Weighs by |
| --- | --- |
| [by_samples][flypse.strategies.weighting.by_samples] | The number of samples in the data of a message ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) |
| [polynomial_staleness][flypse.strategies.weighting.polynomial_staleness] | `(1 + τ)^-a`, with `τ` how many versions behind it is ([Xie et al., 2019](https://arxiv.org/abs/1903.03934)) |
| [hinge_staleness][flypse.strategies.weighting.hinge_staleness] | 1 up to a staleness of `b`, then decaying ([Xie et al., 2019](https://arxiv.org/abs/1903.03934)) |
| [inverse_probability][flypse.strategies.weighting.inverse_probability] | The inverse of the probability to be selected |
| [min_norm][flypse.strategies.weighting.min_norm] | The combination of minimum norm within `ε` of the prior; with `ε = 1` it hurts no contribution ([Hu et al., 2022](https://arxiv.org/abs/2006.11489)) |
| [FedADP][flypse.strategies.weighting.FedADP] | The angle with the aggregate, smoothed over the rounds ([Wu and Wang, 2021](https://arxiv.org/abs/2012.00661)) |
| [FoolsGold][flypse.strategies.weighting.FoolsGold] | How much a history of updates resembles another's ([Fung et al., 2020](https://arxiv.org/abs/1808.04866)) |

[normalise][flypse.strategies.weighting.normalise] multiplies any number of factors and
scales the product to sum to 1:

```python
from flypse.strategies import aggregation, weighting

weights = weighting.normalise(weighting.by_samples(messages))
average = aggregation.mean([m.record.parameters for m in messages], weights)
```

[by_samples][flypse.strategies.weighting.by_samples] reads `n_samples` from the data of
each message. If your clients use another name, pass it as `sample_size_key`.

### By staleness

[lag][flypse.federation.model.FederatedModel.lag] tells how many versions a contribution is
behind the model of the entity, which is all a staleness factor needs:

```python
lags = [self.model.lag(m.record, "global") for m in messages]
weights = weighting.normalise(
    weighting.by_samples(messages),
    weighting.polynomial_staleness(lags),
)
```

### By the probability to be selected

The inverse of the [selection](selection.md#correct-for-the-selection) probabilities removes
the bias a selection introduces. Whether the weights are normalised makes the difference
between two estimators of the mean over *all* the candidates:

=== "Ratio"

    Normalised, the mean is consistent, and usually varies less:

    ```python
    weights = weighting.normalise(
        weighting.by_samples(messages),
        weighting.inverse_probability([pi[m.sender] for m in messages]),
    )
    average = aggregation.mean(vectors, weights)
    ```

=== "Horvitz-Thompson"

    Not normalised, the sum is unbiased. It is an estimator for *updates*: the weights do not
    sum to 1, so the same sum of models is not a model. Each selected candidate counts once,
    every one of them must answer, and the sizes of all the candidates must be known
    beforehand:

    ```python
    total = sum(self.sizes.values())
    shares = np.array([self.sizes[m.sender] / total for m in messages])
    weights = shares * weighting.inverse_probability([pi[m.sender] for m in messages])
    update = np.sum(weights[:, None] * np.vstack(deltas), axis=0)
    ```

### By similarity

[FedADP][flypse.strategies.weighting.FedADP] and
[FoolsGold][flypse.strategies.weighting.FoolsGold] remember the senders, so an entity
creates one and keeps it:

```python
class Server(Entity):
    def __init__(self, entity_id):
        super().__init__(entity_id, with_data=False)
        self.weigher = weighting.FedADP()
        ...

    async def step(self):
        ...
        samples = weighting.by_samples(messages)
        reference = aggregation.mean(deltas, samples)
        angles = self.weigher.weigh([m.sender for m in messages], deltas, reference)
        delta = aggregation.mean(deltas, weighting.normalise(samples, angles))
```

!!! warning

    [FoolsGold][flypse.strategies.weighting.FoolsGold] takes resemblance for collusion. Under
    IID data honest clients resemble each other too, and lose weight.

## One row per peer

With no server, each peer averages its own model with those of its neighbours. Its weights
are a row of the mixing matrix of the graph, computed from the
[connections](../concepts/federation.md#neighbours-and-direction) of the peer:

```python
import numpy as np

connections = await self.get_connections()
own, row = weighting.metropolis_hastings(connections)

await self.bcast(record)
peers = sorted(row)
messages = [await self.receive(source=peer) for peer in peers]

vectors = [record.parameters, *(m.record.parameters for m in messages)]
weights = np.array([own, *(row[peer] for peer in peers)])
self.model.update_version(aggregation.mean(vectors, weights), "global")
```

| Strategy | Weight of a neighbour | Peers agree on |
| --- | --- | --- |
| [metropolis_hastings][flypse.strategies.weighting.metropolis_hastings] | `1 / (1 + max(d_i, d_j))`, with `d` the degrees | The average of their models, on any connected undirected graph |
| [uniform_mixing][flypse.strategies.weighting.uniform_mixing] | `1 / (d_i + 1)` | The average only if all the degrees are the same; otherwise one weighted by `d_i + 1` |

## One row per recipient

Personalised federated learning gives every client a model of its own, made mostly of the
models that resemble its own. These strategies return a matrix: row `i` weighs the
contributions for recipient `i`.

| Strategy | Compares |
| --- | --- |
| [fedamp][flypse.strategies.weighting.fedamp] | Models, by their distance ([Huang et al., 2021](https://arxiv.org/abs/2007.03797)) |
| [heurfedamp][flypse.strategies.weighting.heurfedamp] | Models or updates, by their angle, with a fixed weight for the client's own ([Huang et al., 2021](https://arxiv.org/abs/2007.03797)) |
| [pfedgraph][flypse.strategies.weighting.pfedgraph] | Updates, by their angle, with a pull towards a prior ([Ye et al., 2023](https://proceedings.mlr.press/v202/ye23b.html)) |

Each personalised model is a version of its own, so give the models a level for it, with
`version_levels=("global", "personal", "local")`:

```python
import asyncio

vectors = [m.record.parameters for m in messages]
weights = weighting.heurfedamp(deltas, sigma=10, self_weight=0.3)

self.model.update_version(aggregation.mean(vectors), "global")
records = [
    self.model.update_version(aggregation.mean(vectors, row), "personal")
    for row in weights
]
await asyncio.gather(
    *(self.send(m.sender, r) for m, r in zip(messages, records, strict=True))
)
```

!!! note

    The weights come from the *updates*, `deltas`, and the models are what they combine. Models
    that start a round from the same parameters have cosines that are all close to 1.

!!! note "See also"

    - [Versions](../concepts/versions.md)
    - [Selection](selection.md)
    - [Aggregation](aggregation.md)
